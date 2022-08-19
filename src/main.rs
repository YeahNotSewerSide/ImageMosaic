mod mosaic;
use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, value_parser)]
    tiles: String,

    #[clap(short, long, value_parser)]
    source: String,

    #[clap(short, long, value_parser)]
    output: String,

    #[clap(short, long, value_parser, default_value_t = 20)]
    width: u32,

    #[clap(short, long, value_parser, default_value_t = 20)]
    height: u32,

    #[clap(short, long, value_parser)]
    resize: bool,
}

fn main() {
    let args = Args::parse();

    let kernel_size = (args.width, args.height);

    let tiles = mosaic::prepare_tiles(&args.tiles, kernel_size).unwrap();

    let image = if args.resize {
        mosaic::build_mosaic(&args.source, &tiles, kernel_size).unwrap()
    } else {
        mosaic::build_mosaic_without_compression(&args.source, &tiles, kernel_size).unwrap()
    };

    image.save(args.output).unwrap();
}
