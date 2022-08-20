mod mosaic;
use clap::Parser;

const LONG_ABOUT:&str = "This is a simple generator of image mosaics.";

#[derive(Parser, Debug)]
#[clap(version, about="ImageMosaic generator", long_about = LONG_ABOUT)]
struct Args {
    #[clap(short, long, value_parser)]
    /// A directory with image-tiles for mosaic 
    tiles: String,

    #[clap(short, long, value_parser)]
    /// A source image, from which to produce mosaic with tiles
    source: String,

    #[clap(short, long, value_parser)]
    /// Output file
    output: String,

    #[clap(short, long, value_parser, default_value_t = 20)]
    /// Desired width of tiles
    width: u32,

    #[clap(short, long, value_parser, default_value_t = 20)]
    /// Desired height of tiles
    height: u32,

    #[clap(short, long, value_parser)]
    /// With a resize on, new image will be the size of (source.width*width, source.height*height)
    resize: bool,

    #[clap(long, value_parser)]
    /// Desired opacity of tiles, 0 - min, 255 - max
    opacity: Option<u8>
}

fn main() {
    let args = Args::parse();

    let kernel_size = (args.width, args.height);

    let tiles = mosaic::prepare_tiles(&args.tiles, kernel_size).unwrap();

    let image = if !args.resize {
        match args.opacity{
            Some(opacity) => {
                if opacity == 255 || opacity == 0{
                    mosaic::build_mosaic(&args.source, &tiles, kernel_size).unwrap()    
                }else{
                    mosaic::build_mosaic_blend(&args.source, &tiles, kernel_size, opacity).unwrap()
                }
            },
            None => {
                mosaic::build_mosaic(&args.source, &tiles, kernel_size).unwrap()        
            }
        }
    } else {
        mosaic::build_mosaic_without_compression(&args.source, &tiles, kernel_size).unwrap()
    };

    image.save(args.output).unwrap();
}
