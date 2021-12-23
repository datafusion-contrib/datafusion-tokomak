// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Benchmark derived from TPC-H. This is not an official TPC-H benchmark.

use std::{fs, iter::Iterator, path::{Path, PathBuf}, sync::Arc, time::{Duration, Instant}};

use ballista::context::BallistaContext;
use ballista::prelude::{BallistaConfig, BALLISTA_DEFAULT_SHUFFLE_PARTITIONS};

use datafusion::{datasource::{MemTable, TableProvider}, optimizer::{filter_push_down::FilterPushDown, projection_push_down::ProjectionPushDown, common_subexpr_eliminate::CommonSubexprEliminate}};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_plan::LogicalPlan;
use datafusion::parquet::basic::Compression;
use datafusion::parquet::file::properties::WriterProperties;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::*;
use datafusion::{
    arrow::datatypes::{DataType, Field, Schema},
    datasource::file_format::{csv::CsvFormat, FileFormat},
};
use datafusion::{
    arrow::record_batch::RecordBatch, datasource::file_format::parquet::ParquetFormat,
};
use datafusion::{
    arrow::util::pretty,
    datasource::{
        listing::{ListingOptions, ListingTable},
        object_store::local::LocalFileSystem,
    },
};

use structopt::StructOpt;
use tokomak::{RunnerSettings, Tokomak, ALL_RULES} ;

#[cfg(feature = "snmalloc")]
#[global_allocator]
static ALLOC: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, StructOpt, Clone)]
struct BallistaBenchmarkOpt {
    /// Query number
    #[structopt(short, long)]
    query: usize,

    /// Activate debug mode to see query results
    #[structopt(short, long)]
    debug: bool,

    /// Number of iterations of each test run
    #[structopt(short = "i", long = "iterations", default_value = "3")]
    iterations: usize,

    // /// Batch size when reading CSV or Parquet files
    // #[structopt(short = "s", long = "batch-size", default_value = "8192")]
    // batch_size: usize,
    /// Path to data files
    #[structopt(parse(from_os_str), required = true, short = "z", long = "data")]
    path: PathBuf,

    /// File format: `csv` or `parquet`
    #[structopt(short = "f", long = "format", default_value = "csv")]
    file_format: String,

    // /// Load the data into a MemTable before executing the query
    // #[structopt(short = "m", long = "mem-table")]
    // mem_table: bool,
    /// Number of partitions to process in parallel
    #[structopt(short = "n", long = "partitions", default_value = "2")]
    partitions: usize,

    /// Ballista executor host
    #[structopt(long = "host")]
    host: Option<String>,

    /// Ballista executor port
    #[structopt(long = "port")]
    port: Option<u16>,
}

#[derive(Debug, StructOpt, Clone)]
struct DataFusionBenchmarkOpt {
    /// Query number
    #[structopt(short, long)]
    query: usize,

    /// Activate debug mode to see query results
    #[structopt(short, long)]
    debug: bool,

    /// Number of iterations of each test run
    #[structopt(short = "i", long = "iterations", default_value = "3")]
    iterations: usize,

    /// Number of partitions to process in parallel
    #[structopt(short = "n", long = "partitions", default_value = "2")]
    partitions: usize,

    /// Batch size when reading CSV or Parquet files
    #[structopt(short = "s", long = "batch-size", default_value = "8192")]
    batch_size: usize,

    /// Path to data files
    #[structopt(parse(from_os_str), required = true,  long = "data")]
    path: PathBuf,

    /// File format: `csv` or `parquet`
    #[structopt(short = "f", long = "format", default_value = "csv")]
    file_format: String,

    /// Load the data into a MemTable before executing the query
    #[structopt(short = "m", long = "mem-table")]
    mem_table: bool,

    #[structopt(short="t", long="tokomak")]
    tokomak:bool
}

#[derive(Debug, StructOpt)]
struct ConvertOpt {
    /// Path to csv files
    #[structopt(parse(from_os_str), required = true, short = "i", long = "input")]
    input_path: PathBuf,

    /// Output path
    #[structopt(parse(from_os_str), required = true, short = "o", long = "output")]
    output_path: PathBuf,

    /// Output file format: `csv` or `parquet`
    #[structopt(short = "f", long = "format")]
    file_format: String,

    /// Compression to use when writing Parquet files
    #[structopt(short = "c", long = "compression", default_value = "zstd")]
    compression: String,

    /// Number of partitions to produce
    #[structopt(short = "n", long = "partitions", default_value = "1")]
    partitions: usize,

    /// Batch size when reading CSV or Parquet files
    #[structopt(short = "s", long = "batch-size", default_value = "8192")]
    batch_size: usize,
}

#[derive(Debug, StructOpt)]
#[structopt(about = "benchmark command")]
enum BenchmarkSubCommandOpt {
    #[structopt(name = "ballista")]
    BallistaBenchmark(BallistaBenchmarkOpt),
    #[structopt(name = "datafusion")]
    DataFusionBenchmark(DataFusionBenchmarkOpt),
}

#[derive(Debug, StructOpt)]
#[structopt(name = "TPC-H", about = "TPC-H Benchmarks.")]
enum TpchOpt {
    Benchmark(BenchmarkSubCommandOpt),
    Convert(ConvertOpt),
}

const TABLES: &[&str] = &[
    "part", "supplier", "partsupp", "customer", "orders", "lineitem", "nation", "region",
];

#[tokio::main]
async fn main() -> Result<()> {
    use BenchmarkSubCommandOpt::*;

    env_logger::init();
    match TpchOpt::from_args() {
        TpchOpt::Benchmark(BallistaBenchmark(opt)) => {
            benchmark_ballista(opt).await.map(|_| ())
        }
        TpchOpt::Benchmark(DataFusionBenchmark(opt)) => {
            benchmark_datafusion(opt).await.map(|_| ())
        }
        TpchOpt::Convert(opt) => convert_tbl(opt).await,
    }
}

fn get_tokomak_opt_seconds()->f64{
    const DEFAULT_TIME: f64 = 0.5;
    let str_time = std::env::var("TOKOMAK_OPT_TIME").unwrap_or_else(|_| format!("{}",DEFAULT_TIME));
    let opt_seconds: f64 = str_time.parse().unwrap_or(DEFAULT_TIME);
    println!("Tokomak optimizer will run for {}s",opt_seconds);
    opt_seconds 
}


async fn benchmark_datafusion(opt: DataFusionBenchmarkOpt) -> Result<Vec<RecordBatch>> {
    println!("Running benchmarks with the following options: {:?}", opt);
    let mut settings = RunnerSettings::new();
    println!("Setup up runner settings");
    settings
        .with_iter_limit(1000)
        .with_node_limit(1_000_000)
        .with_time_limit(Duration::from_secs_f64(get_tokomak_opt_seconds()));
    let tokomak_optimizer = Tokomak::with_builtin_rules(settings, ALL_RULES);
    let mut config = ExecutionConfig::new()
        .with_target_partitions(opt.partitions)
        .with_batch_size(opt.batch_size);
    if opt.tokomak{
        println!("Adding tokomak optimizer");
        config = config.with_optimizer_rules(vec![
                Arc::new(tokomak_optimizer), 
                Arc::new(CommonSubexprEliminate::new()),
                Arc::new(ProjectionPushDown::new()),
                Arc::new(FilterPushDown::new()),
               
            ]);
    }else{
        println!("Did not add tokomak optimizer");
    }

    let mut ctx = ExecutionContext::with_config(config);

    // register tables
    for table in TABLES {
        let table_provider = get_table(
            opt.path.to_str().unwrap(),
            table,
            opt.file_format.as_str(),
            opt.partitions,
        )?;
        if opt.mem_table {
            println!("Loading table '{}' into memory", table);
            let start = Instant::now();

            let memtable =
                MemTable::load(table_provider, opt.batch_size, Some(opt.partitions))
                    .await?;
            println!(
                "Loaded table '{}' into memory in {} ms",
                table,
                start.elapsed().as_millis()
            );
            ctx.register_table(*table, Arc::new(memtable))?;
        } else {
            ctx.register_table(*table, table_provider)?;
        }
    }

    let mut millis = vec![];
    // run benchmark
    let mut result: Vec<RecordBatch> = Vec::with_capacity(1);
    
    for i in 0..opt.iterations {
        println!("Startin iteration {}", i);
        let plan = create_logical_plan(&mut ctx, opt.query)?;
        
        let start = Instant::now();
        result = execute_query(&mut ctx, &plan, opt.debug).await?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        millis.push(elapsed as f64);
        println!("Query {} iteration {} took {:.1} ms", opt.query, i, elapsed);
    }

    let avg = millis.iter().sum::<f64>() / millis.len() as f64;
    println!("Query {} avg time: {:.2} ms", opt.query, avg);

    Ok(result)
}

async fn benchmark_ballista(opt: BallistaBenchmarkOpt) -> Result<()> {
    println!("Running benchmarks with the following options: {:?}", opt);

    let config = BallistaConfig::builder()
        .set(
            BALLISTA_DEFAULT_SHUFFLE_PARTITIONS,
            &format!("{}", opt.partitions),
        )
        .build()
        .map_err(|e| DataFusionError::Execution(format!("{:?}", e)))?;

    let ctx =
        BallistaContext::remote(opt.host.unwrap().as_str(), opt.port.unwrap(), &config);

    // register tables with Ballista context
    let path = opt.path.to_str().unwrap();
    let file_format = opt.file_format.as_str();
    for table in TABLES {
        match file_format {
            // dbgen creates .tbl ('|' delimited) files without header
            "tbl" => {
                let path = format!("{}/{}.tbl", path, table);
                let schema = get_schema(table);
                let options = CsvReadOptions::new()
                    .schema(&schema)
                    .delimiter(b'|')
                    .has_header(false)
                    .file_extension(".tbl");
                ctx.register_csv(table, &path, options)
                    .await
                    .map_err(|e| DataFusionError::Plan(format!("{:?}", e)))?;
            }
            "csv" => {
                let path = format!("{}/{}", path, table);
                let schema = get_schema(table);
                let options = CsvReadOptions::new().schema(&schema).has_header(true);
                ctx.register_csv(table, &path, options)
                    .await
                    .map_err(|e| DataFusionError::Plan(format!("{:?}", e)))?;
            }
            "parquet" => {
                let path = format!("{}/{}", path, table);
                ctx.register_parquet(table, &path)
                    .await
                    .map_err(|e| DataFusionError::Plan(format!("{:?}", e)))?;
            }
            other => {
                unimplemented!("Invalid file format '{}'", other);
            }
        }
    }

    let mut millis = vec![];

    // run benchmark
    let sql = get_query_sql(opt.query)?;
    println!("Running benchmark with query {}:\n {}", opt.query, sql);
    for i in 0..opt.iterations {
        let start = Instant::now();
        let df = ctx
            .sql(&sql)
            .await
            .map_err(|e| DataFusionError::Plan(format!("{:?}", e)))?;
        let batches = df
            .collect()
            .await
            .map_err(|e| DataFusionError::Plan(format!("{:?}", e)))?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        millis.push(elapsed as f64);
        println!("Query {} iteration {} took {:.1} ms", opt.query, i, elapsed);
        if opt.debug {
            pretty::print_batches(&batches)?;
        }
    }

    let avg = millis.iter().sum::<f64>() / millis.len() as f64;
    println!("Query {} avg time: {:.2} ms", opt.query, avg);

    Ok(())
}

fn get_query_sql(query: usize) -> Result<String> {
    if query > 0 && query < 23 {
        let filename = format!("/home/patrick/dev/rust/arrow-datafusion/benchmarks/queries/q{}.sql", query);
        Ok(fs::read_to_string(&filename).expect("failed to read query"))
    } else {
        Err(DataFusionError::Plan(
            "invalid query. Expected value between 1 and 22".to_owned(),
        ))
    }
}

fn create_logical_plan(ctx: &mut ExecutionContext, query: usize) -> Result<LogicalPlan> {
    let sql = get_query_sql(query)?;
    ctx.create_logical_plan(&sql)
}

async fn execute_query(
    ctx: &mut ExecutionContext,
    plan: &LogicalPlan,
    debug: bool,
) -> Result<Vec<RecordBatch>> {
    if debug {
        println!("=== Logical plan ===\n{:?}\n", plan);
    }
    let start = Instant::now();
    let plan = ctx.optimize(plan)?;
    let elapsed = start.elapsed().as_secs_f64()*1000.0;
    println!("Spent {:.2} ms optimizing", elapsed);
    if debug {
        println!("=== Optimized logical plan ===\n{:?}\n", plan);
    }
    //std::process::exit(0);
    let physical_plan = ctx.create_physical_plan(&plan).await?;
    if debug {
        println!(
            "=== Physical plan ===\n{}\n",
            displayable(physical_plan.as_ref()).indent().to_string()
        );
    }
    let result = collect(physical_plan.clone()).await?;
    if debug {
        println!(
            "=== Physical plan with metrics ===\n{}\n",
            DisplayableExecutionPlan::with_metrics(physical_plan.as_ref())
                .indent()
                .to_string()
        );
        pretty::print_batches(&result)?;
    }
    Ok(result)
}

async fn convert_tbl(opt: ConvertOpt) -> Result<()> {
    let output_root_path = Path::new(&opt.output_path);
    for table in TABLES {
        let start = Instant::now();
        let schema = get_schema(table);

        let input_path = format!("{}/{}.tbl", opt.input_path.to_str().unwrap(), table);
        let options = CsvReadOptions::new()
            .schema(&schema)
            .delimiter(b'|')
            .file_extension(".tbl");

        let config = ExecutionConfig::new().with_batch_size(opt.batch_size);
        let mut ctx = ExecutionContext::with_config(config);

        // build plan to read the TBL file
        let mut csv = ctx.read_csv(&input_path, options).await?;

        // optionally, repartition the file
        if opt.partitions > 1 {
            csv = csv.repartition(Partitioning::RoundRobinBatch(opt.partitions))?
        }

        // create the physical plan
        let csv = csv.to_logical_plan();
        let csv = ctx.optimize(&csv)?;
        let csv = ctx.create_physical_plan(&csv).await?;

        let output_path = output_root_path.join(table);
        let output_path = output_path.to_str().unwrap().to_owned();

        println!(
            "Converting '{}' to {} files in directory '{}'",
            &input_path, &opt.file_format, &output_path
        );
        match opt.file_format.as_str() {
            "csv" => ctx.write_csv(csv, output_path).await?,
            "parquet" => {
                let compression = match opt.compression.as_str() {
                    "none" => Compression::UNCOMPRESSED,
                    "snappy" => Compression::SNAPPY,
                    "brotli" => Compression::BROTLI,
                    "gzip" => Compression::GZIP,
                    "lz4" => Compression::LZ4,
                    "lz0" => Compression::LZO,
                    "zstd" => Compression::ZSTD,
                    other => {
                        return Err(DataFusionError::NotImplemented(format!(
                            "Invalid compression format: {}",
                            other
                        )))
                    }
                };
                let props = WriterProperties::builder()
                    .set_compression(compression)
                    .build();
                ctx.write_parquet(csv, output_path, Some(props)).await?
            }
            other => {
                return Err(DataFusionError::NotImplemented(format!(
                    "Invalid output format: {}",
                    other
                )))
            }
        }
        println!("Conversion completed in {} ms", start.elapsed().as_millis());
    }

    Ok(())
}

fn get_table(
    path: &str,
    table: &str,
    table_format: &str,
    target_partitions: usize,
) -> Result<Arc<dyn TableProvider>> {
    let (format, path, extension): (Arc<dyn FileFormat>, String, &'static str) =
        match table_format {
            // dbgen creates .tbl ('|' delimited) files without header
            "tbl" => {
                let path = format!("{}/{}.tbl", path, table);

                let format = CsvFormat::default()
                    .with_delimiter(b'|')
                    .with_has_header(false);

                (Arc::new(format), path, ".tbl")
            }
            "csv" => {
                let path = format!("{}/{}", path, table);
                let format = CsvFormat::default()
                    .with_delimiter(b',')
                    .with_has_header(true);

                (Arc::new(format), path, ".csv")
            }
            "parquet" => {
                let path = format!("{}/{}", path, table);
                let format = ParquetFormat::default().with_enable_pruning(true);

                (Arc::new(format), path, ".parquet")
            }
            other => {
                unimplemented!("Invalid file format '{}'", other);
            }
        };
    let schema = Arc::new(get_schema(table));

    let options = ListingOptions {
        format,
        file_extension: extension.to_owned(),
        target_partitions,
        collect_stat: true,
        table_partition_cols: vec![],
    };

    Ok(Arc::new(ListingTable::new(
        Arc::new(LocalFileSystem {}),
        path,
        schema,
        options,
    )))
}

fn get_schema(table: &str) -> Schema {
    // note that the schema intentionally uses signed integers so that any generated Parquet
    // files can also be used to benchmark tools that only support signed integers, such as
    // Apache Spark

    match table {
        "part" => Schema::new(vec![
            Field::new("p_partkey", DataType::Int64, false),
            Field::new("p_name", DataType::Utf8, false),
            Field::new("p_mfgr", DataType::Utf8, false),
            Field::new("p_brand", DataType::Utf8, false),
            Field::new("p_type", DataType::Utf8, false),
            Field::new("p_size", DataType::Int32, false),
            Field::new("p_container", DataType::Utf8, false),
            Field::new("p_retailprice", DataType::Float64, false),
            Field::new("p_comment", DataType::Utf8, false),
        ]),

        "supplier" => Schema::new(vec![
            Field::new("s_suppkey", DataType::Int64, false),
            Field::new("s_name", DataType::Utf8, false),
            Field::new("s_address", DataType::Utf8, false),
            Field::new("s_nationkey", DataType::Int64, false),
            Field::new("s_phone", DataType::Utf8, false),
            Field::new("s_acctbal", DataType::Float64, false),
            Field::new("s_comment", DataType::Utf8, false),
        ]),

        "partsupp" => Schema::new(vec![
            Field::new("ps_partkey", DataType::Int64, false),
            Field::new("ps_suppkey", DataType::Int64, false),
            Field::new("ps_availqty", DataType::Int32, false),
            Field::new("ps_supplycost", DataType::Float64, false),
            Field::new("ps_comment", DataType::Utf8, false),
        ]),

        "customer" => Schema::new(vec![
            Field::new("c_custkey", DataType::Int64, false),
            Field::new("c_name", DataType::Utf8, false),
            Field::new("c_address", DataType::Utf8, false),
            Field::new("c_nationkey", DataType::Int64, false),
            Field::new("c_phone", DataType::Utf8, false),
            Field::new("c_acctbal", DataType::Float64, false),
            Field::new("c_mktsegment", DataType::Utf8, false),
            Field::new("c_comment", DataType::Utf8, false),
        ]),

        "orders" => Schema::new(vec![
            Field::new("o_orderkey", DataType::Int64, false),
            Field::new("o_custkey", DataType::Int64, false),
            Field::new("o_orderstatus", DataType::Utf8, false),
            Field::new("o_totalprice", DataType::Float64, false),
            Field::new("o_orderdate", DataType::Date32, false),
            Field::new("o_orderpriority", DataType::Utf8, false),
            Field::new("o_clerk", DataType::Utf8, false),
            Field::new("o_shippriority", DataType::Int32, false),
            Field::new("o_comment", DataType::Utf8, false),
        ]),

        "lineitem" => Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, false),
            Field::new("l_partkey", DataType::Int64, false),
            Field::new("l_suppkey", DataType::Int64, false),
            Field::new("l_linenumber", DataType::Int32, false),
            Field::new("l_quantity", DataType::Float64, false),
            Field::new("l_extendedprice", DataType::Float64, false),
            Field::new("l_discount", DataType::Float64, false),
            Field::new("l_tax", DataType::Float64, false),
            Field::new("l_returnflag", DataType::Utf8, false),
            Field::new("l_linestatus", DataType::Utf8, false),
            Field::new("l_shipdate", DataType::Date32, false),
            Field::new("l_commitdate", DataType::Date32, false),
            Field::new("l_receiptdate", DataType::Date32, false),
            Field::new("l_shipinstruct", DataType::Utf8, false),
            Field::new("l_shipmode", DataType::Utf8, false),
            Field::new("l_comment", DataType::Utf8, false),
        ]),

        "nation" => Schema::new(vec![
            Field::new("n_nationkey", DataType::Int64, false),
            Field::new("n_name", DataType::Utf8, false),
            Field::new("n_regionkey", DataType::Int64, false),
            Field::new("n_comment", DataType::Utf8, false),
        ]),

        "region" => Schema::new(vec![
            Field::new("r_regionkey", DataType::Int64, false),
            Field::new("r_name", DataType::Utf8, false),
            Field::new("r_comment", DataType::Utf8, false),
        ]),

        _ => unimplemented!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Arc;

    use datafusion::arrow::array::*;
    use datafusion::arrow::util::display::array_value_to_string;
    use datafusion::logical_plan::Expr;
    use datafusion::logical_plan::Expr::Cast;

    #[tokio::test]
    async fn q1() -> Result<()> {
        verify_query(1).await
    }

    #[tokio::test]
    async fn q2() -> Result<()> {
        verify_query(2).await
    }

    #[tokio::test]
    async fn q3() -> Result<()> {
        verify_query(3).await
    }

    #[tokio::test]
    async fn q4() -> Result<()> {
        verify_query(4).await
    }

    #[tokio::test]
    async fn q5() -> Result<()> {
        verify_query(5).await
    }

    #[tokio::test]
    async fn q6() -> Result<()> {
        verify_query(6).await
    }

    #[tokio::test]
    async fn q7() -> Result<()> {
        verify_query(7).await
    }

    #[tokio::test]
    async fn q8() -> Result<()> {
        verify_query(8).await
    }

    #[tokio::test]
    async fn q9() -> Result<()> {
        verify_query(9).await
    }

    #[tokio::test]
    async fn q10() -> Result<()> {
        verify_query(10).await
    }

    #[tokio::test]
    async fn q11() -> Result<()> {
        verify_query(11).await
    }

    #[tokio::test]
    async fn q12() -> Result<()> {
        verify_query(12).await
    }

    #[tokio::test]
    async fn q13() -> Result<()> {
        verify_query(13).await
    }

    #[tokio::test]
    async fn q14() -> Result<()> {
        verify_query(14).await
    }

    #[tokio::test]
    async fn q15() -> Result<()> {
        verify_query(15).await
    }

    #[tokio::test]
    async fn q16() -> Result<()> {
        verify_query(16).await
    }

    #[tokio::test]
    async fn q17() -> Result<()> {
        verify_query(17).await
    }

    #[tokio::test]
    async fn q18() -> Result<()> {
        verify_query(18).await
    }

    #[tokio::test]
    async fn q19() -> Result<()> {
        verify_query(19).await
    }

    #[tokio::test]
    async fn q20() -> Result<()> {
        verify_query(20).await
    }

    #[tokio::test]
    async fn q21() -> Result<()> {
        verify_query(21).await
    }

    #[tokio::test]
    async fn q22() -> Result<()> {
        verify_query(22).await
    }

    #[tokio::test]
    async fn run_q1() -> Result<()> {
        run_query(1).await
    }

    #[tokio::test]
    async fn run_q3() -> Result<()> {
        run_query(3).await
    }

    #[tokio::test]
    async fn run_q5() -> Result<()> {
        run_query(5).await
    }

    #[tokio::test]
    async fn run_q6() -> Result<()> {
        run_query(6).await
    }

    #[tokio::test]
    async fn run_q7() -> Result<()> {
        run_query(7).await
    }

    #[tokio::test]
    async fn run_q8() -> Result<()> {
        run_query(8).await
    }

    #[tokio::test]
    async fn run_q9() -> Result<()> {
        run_query(9).await
    }

    #[tokio::test]
    async fn run_q10() -> Result<()> {
        run_query(10).await
    }

    #[tokio::test]
    async fn run_q12() -> Result<()> {
        run_query(12).await
    }

    #[tokio::test]
    async fn run_q13() -> Result<()> {
        run_query(13).await
    }

    #[tokio::test]
    async fn run_q14() -> Result<()> {
        run_query(14).await
    }

    #[tokio::test]
    async fn run_q19() -> Result<()> {
        run_query(19).await
    }

    /// Specialised String representation
    fn col_str(column: &ArrayRef, row_index: usize) -> String {
        if column.is_null(row_index) {
            return "NULL".to_string();
        }

        // Special case ListArray as there is no pretty print support for it yet
        if let DataType::FixedSizeList(_, n) = column.data_type() {
            let array = column
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap()
                .value(row_index);

            let mut r = Vec::with_capacity(*n as usize);
            for i in 0..*n {
                r.push(col_str(&array, i as usize));
            }
            return format!("[{}]", r.join(","));
        }

        array_value_to_string(column, row_index).unwrap()
    }

    /// Converts the results into a 2d array of strings, `result[row][column]`
    /// Special cases nulls to NULL for testing
    fn result_vec(results: &[RecordBatch]) -> Vec<Vec<String>> {
        let mut result = vec![];
        for batch in results {
            for row_index in 0..batch.num_rows() {
                let row_vec = batch
                    .columns()
                    .iter()
                    .map(|column| col_str(column, row_index))
                    .collect();
                result.push(row_vec);
            }
        }
        result
    }

    fn get_answer_schema(n: usize) -> Schema {
        match n {
            1 => Schema::new(vec![
                Field::new("l_returnflag", DataType::Utf8, true),
                Field::new("l_linestatus", DataType::Utf8, true),
                Field::new("sum_qty", DataType::Float64, true),
                Field::new("sum_base_price", DataType::Float64, true),
                Field::new("sum_disc_price", DataType::Float64, true),
                Field::new("sum_charge", DataType::Float64, true),
                Field::new("avg_qty", DataType::Float64, true),
                Field::new("avg_price", DataType::Float64, true),
                Field::new("avg_disc", DataType::Float64, true),
                Field::new("count_order", DataType::UInt64, true),
            ]),

            2 => Schema::new(vec![
                Field::new("s_acctbal", DataType::Float64, true),
                Field::new("s_name", DataType::Utf8, true),
                Field::new("n_name", DataType::Utf8, true),
                Field::new("p_partkey", DataType::Int32, true),
                Field::new("p_mfgr", DataType::Utf8, true),
                Field::new("s_address", DataType::Utf8, true),
                Field::new("s_phone", DataType::Utf8, true),
                Field::new("s_comment", DataType::Utf8, true),
            ]),

            3 => Schema::new(vec![
                Field::new("l_orderkey", DataType::Int32, true),
                Field::new("revenue", DataType::Float64, true),
                Field::new("o_orderdate", DataType::Date32, true),
                Field::new("o_shippriority", DataType::Int32, true),
            ]),

            4 => Schema::new(vec![
                Field::new("o_orderpriority", DataType::Utf8, true),
                Field::new("order_count", DataType::Int32, true),
            ]),

            5 => Schema::new(vec![
                Field::new("n_name", DataType::Utf8, true),
                Field::new("revenue", DataType::Float64, true),
            ]),

            6 => Schema::new(vec![Field::new("revenue", DataType::Float64, true)]),

            7 => Schema::new(vec![
                Field::new("supp_nation", DataType::Utf8, true),
                Field::new("cust_nation", DataType::Utf8, true),
                Field::new("l_year", DataType::Int32, true),
                Field::new("revenue", DataType::Float64, true),
            ]),

            8 => Schema::new(vec![
                Field::new("o_year", DataType::Int32, true),
                Field::new("mkt_share", DataType::Float64, true),
            ]),

            9 => Schema::new(vec![
                Field::new("nation", DataType::Utf8, true),
                Field::new("o_year", DataType::Int32, true),
                Field::new("sum_profit", DataType::Float64, true),
            ]),

            10 => Schema::new(vec![
                Field::new("c_custkey", DataType::Int32, true),
                Field::new("c_name", DataType::Utf8, true),
                Field::new("revenue", DataType::Float64, true),
                Field::new("c_acctbal", DataType::Float64, true),
                Field::new("n_name", DataType::Utf8, true),
                Field::new("c_address", DataType::Utf8, true),
                Field::new("c_phone", DataType::Utf8, true),
                Field::new("c_comment", DataType::Utf8, true),
            ]),

            11 => Schema::new(vec![
                Field::new("ps_partkey", DataType::Int32, true),
                Field::new("value", DataType::Float64, true),
            ]),

            12 => Schema::new(vec![
                Field::new("l_shipmode", DataType::Utf8, true),
                Field::new("high_line_count", DataType::Int64, true),
                Field::new("low_line_count", DataType::Int64, true),
            ]),

            13 => Schema::new(vec![
                Field::new("c_count", DataType::Int64, true),
                Field::new("custdist", DataType::Int64, true),
            ]),

            14 => Schema::new(vec![Field::new("promo_revenue", DataType::Float64, true)]),

            15 => Schema::new(vec![Field::new("promo_revenue", DataType::Float64, true)]),

            16 => Schema::new(vec![
                Field::new("p_brand", DataType::Utf8, true),
                Field::new("p_type", DataType::Utf8, true),
                Field::new("c_phone", DataType::Int32, true),
                Field::new("c_comment", DataType::Int32, true),
            ]),

            17 => Schema::new(vec![Field::new("avg_yearly", DataType::Float64, true)]),

            18 => Schema::new(vec![
                Field::new("c_name", DataType::Utf8, true),
                Field::new("c_custkey", DataType::Int32, true),
                Field::new("o_orderkey", DataType::Int32, true),
                Field::new("o_orderdate", DataType::Date32, true),
                Field::new("o_totalprice", DataType::Float64, true),
                Field::new("sum_l_quantity", DataType::Float64, true),
            ]),

            19 => Schema::new(vec![Field::new("revenue", DataType::Float64, true)]),

            20 => Schema::new(vec![
                Field::new("s_name", DataType::Utf8, true),
                Field::new("s_address", DataType::Utf8, true),
            ]),

            21 => Schema::new(vec![
                Field::new("s_name", DataType::Utf8, true),
                Field::new("numwait", DataType::Int32, true),
            ]),

            22 => Schema::new(vec![
                Field::new("cntrycode", DataType::Int32, true),
                Field::new("numcust", DataType::Int32, true),
                Field::new("totacctbal", DataType::Float64, true),
            ]),

            _ => unimplemented!(),
        }
    }

    // convert expected schema to all utf8 so columns can be read as strings to be parsed separately
    // this is due to the fact that the csv parser cannot handle leading/trailing spaces
    fn string_schema(schema: Schema) -> Schema {
        Schema::new(
            schema
                .fields()
                .iter()
                .map(|field| {
                    Field::new(
                        Field::name(field),
                        DataType::Utf8,
                        Field::is_nullable(field),
                    )
                })
                .collect::<Vec<Field>>(),
        )
    }

    // convert the schema to the same but with all columns set to nullable=true.
    // this allows direct schema comparison ignoring nullable.
    fn nullable_schema(schema: Arc<Schema>) -> Schema {
        Schema::new(
            schema
                .fields()
                .iter()
                .map(|field| {
                    Field::new(
                        Field::name(field),
                        Field::data_type(field).to_owned(),
                        true,
                    )
                })
                .collect::<Vec<Field>>(),
        )
    }

    async fn run_query(n: usize) -> Result<()> {
        // Tests running query with empty tables, to see whether they run succesfully.

        let config = ExecutionConfig::new()
            .with_target_partitions(1)
            .with_batch_size(10);
        let mut ctx = ExecutionContext::with_config(config);

        for &table in TABLES {
            let schema = get_schema(table);
            let batch = RecordBatch::new_empty(Arc::new(schema.to_owned()));

            let provider = MemTable::try_new(Arc::new(schema), vec![vec![batch]])?;

            ctx.register_table(table, Arc::new(provider))?;
        }

        let plan = create_logical_plan(&mut ctx, n)?;
        execute_query(&mut ctx, &plan, false).await?;

        Ok(())
    }

    async fn verify_query(n: usize) -> Result<()> {
        if let Ok(path) = env::var("TPCH_DATA") {
            // load expected answers from tpch-dbgen
            // read csv as all strings, trim and cast to expected type as the csv string
            // to value parser does not handle data with leading/trailing spaces
            let mut ctx = ExecutionContext::new();
            let schema = string_schema(get_answer_schema(n));
            let options = CsvReadOptions::new()
                .schema(&schema)
                .delimiter(b'|')
                .file_extension(".out");
            let df = ctx
                .read_csv(&format!("{}/answers/q{}.out", path, n), options)
                .await?;
            let df = df.select(
                get_answer_schema(n)
                    .fields()
                    .iter()
                    .map(|field| {
                        Expr::Alias(
                            Box::new(Cast {
                                expr: Box::new(trim(col(Field::name(field)))),
                                data_type: Field::data_type(field).to_owned(),
                            }),
                            Field::name(field).to_string(),
                        )
                    })
                    .collect::<Vec<Expr>>(),
            )?;
            let expected = df.collect().await?;

            // run the query to compute actual results of the query
            let opt = DataFusionBenchmarkOpt {
                query: n,
                debug: false,
                iterations: 1,
                partitions: 2,
                batch_size: 8192,
                path: PathBuf::from(path.to_string()),
                file_format: "tbl".to_string(),
                mem_table: false,
                tokomak: false
            };
            let actual = benchmark_datafusion(opt).await?;

            // assert schema equality without comparing nullable values
            assert_eq!(
                nullable_schema(expected[0].schema()),
                nullable_schema(actual[0].schema())
            );

            // convert both datasets to Vec<Vec<String>> for simple comparison
            let expected_vec = result_vec(&expected);
            let actual_vec = result_vec(&actual);

            // basic result comparison
            assert_eq!(expected_vec.len(), actual_vec.len());

            // compare each row. this works as all TPC-H queries have determinisically ordered results
            for i in 0..actual_vec.len() {
                assert_eq!(expected_vec[i], actual_vec[i]);
            }
        } else {
            println!("TPCH_DATA environment variable not set, skipping test");
        }

        Ok(())
    }

    mod ballista_round_trip {
        use super::*;
        use ballista_core::serde::protobuf;
        use datafusion::physical_plan::ExecutionPlan;
        use std::convert::TryInto;

        async fn round_trip_query(n: usize) -> Result<()> {
            let config = ExecutionConfig::new()
                .with_target_partitions(1)
                .with_batch_size(10);
            let mut ctx = ExecutionContext::with_config(config);

            // set tpch_data_path to dummy value and skip physical plan serde test when TPCH_DATA
            // is not set.
            let tpch_data_path =
                env::var("TPCH_DATA").unwrap_or_else(|_| "./".to_string());

            for &table in TABLES {
                let schema = get_schema(table);
                let options = CsvReadOptions::new()
                    .schema(&schema)
                    .delimiter(b'|')
                    .has_header(false)
                    .file_extension(".tbl");
                let listing_options = options.to_listing_options(1);
                let provider = ListingTable::new(
                    Arc::new(LocalFileSystem {}),
                    format!("{}/{}.tbl", tpch_data_path, table),
                    Arc::new(schema),
                    listing_options,
                );
                ctx.register_table(table, Arc::new(provider))?;
            }

            // test logical plan round trip
            let plan = create_logical_plan(&mut ctx, n)?;
            let proto: protobuf::LogicalPlanNode = (&plan).try_into().unwrap();
            let round_trip: LogicalPlan = (&proto).try_into().unwrap();
            assert_eq!(
                format!("{:?}", plan),
                format!("{:?}", round_trip),
                "logical plan round trip failed"
            );

            // test optimized logical plan round trip
            let plan = ctx.optimize(&plan)?;
            let proto: protobuf::LogicalPlanNode = (&plan).try_into().unwrap();
            let round_trip: LogicalPlan = (&proto).try_into().unwrap();
            assert_eq!(
                format!("{:?}", plan),
                format!("{:?}", round_trip),
                "optimized logical plan round trip failed"
            );

            // test physical plan roundtrip
            if env::var("TPCH_DATA").is_ok() {
                let physical_plan = ctx.create_physical_plan(&plan).await?;
                let proto: protobuf::PhysicalPlanNode =
                    (physical_plan.clone()).try_into().unwrap();
                let round_trip: Arc<dyn ExecutionPlan> = (&proto).try_into().unwrap();
                assert_eq!(
                    format!("{:?}", physical_plan),
                    format!("{:?}", round_trip),
                    "physical plan round trip failed"
                );
            }

            Ok(())
        }

        macro_rules! test_round_trip {
            ($tn:ident, $query:expr) => {
                #[tokio::test]
                async fn $tn() -> Result<()> {
                    round_trip_query($query).await
                }
            };
        }

        test_round_trip!(q1, 1);
        test_round_trip!(q3, 3);
        test_round_trip!(q5, 5);
        test_round_trip!(q6, 6);
        test_round_trip!(q7, 7);
        test_round_trip!(q8, 8);
        test_round_trip!(q9, 9);
        test_round_trip!(q10, 10);
        test_round_trip!(q12, 12);
        test_round_trip!(q13, 13);
    }
}
