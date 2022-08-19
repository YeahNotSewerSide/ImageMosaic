use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImage, GenericImageView};
use std::fs;
use std::path::Path;

use kdtree::distance::squared_euclidean;

pub type ResultSmall<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn prepare_tiles(dir: &str, kernel_size: (u32, u32)) -> ResultSmall<Vec<DynamicImage>> {
    let dir_path = Path::new(dir);
    let files = fs::read_dir(dir_path)?;

    let mut to_return: Vec<DynamicImage> = Vec::with_capacity(files.count());

    let files = fs::read_dir(dir_path)?;
    for file in files {
        let img = match ImageReader::open(file?.path()) {
            Ok(r) => {
                let img = r.decode();
                if img.is_err() {
                    continue;
                }
                unsafe { img.unwrap_unchecked() }
            }
            Err(_) => {
                continue;
            }
        };

        let resized = img.resize_exact(kernel_size.0, kernel_size.1, FilterType::Nearest);

        let tile = DynamicImage::from(resized.to_rgb8());

        to_return.push(tile);
    }

    Ok(to_return)
}

pub fn mean_color(src: &DynamicImage) -> [f64; 3] {
    let mut to_return = [0f64; 3];

    for (_, _, pixel) in src.pixels() {
        to_return[0] += pixel.0[0] as f64;
        to_return[1] += pixel.0[1] as f64;
        to_return[2] += pixel.0[2] as f64;
    }

    let pixels_amount = (src.width() * src.height()) as f64;
    to_return[0] /= pixels_amount;
    to_return[1] /= pixels_amount;
    to_return[2] /= pixels_amount;

    to_return
}

pub fn build_mosaic(
    src_image: &str,
    tiles: &[DynamicImage],
    kernel_size: (u32, u32),
) -> ResultSmall<DynamicImage> {
    let source = ImageReader::open(src_image)?.decode()?;

    let mut to_return = DynamicImage::new_rgb8(source.width(), source.height());

    let new_width = (source.width() as f64 / kernel_size.0 as f64).round() as u32;
    let new_height = (source.height() as f64 / kernel_size.1 as f64).round() as u32;
    let source = source.resize_exact(new_width, new_height, FilterType::Nearest);

    let mut tree: kdtree::KdTree<f64, usize, [f64; 3]> = kdtree::KdTree::new(3);

    for (index, tile) in tiles.iter().enumerate() {
        let color = mean_color(tile);

        tree.add(color, index)?;
    }

    for (x, y, pixel) in source.pixels() {
        let pxl = [pixel.0[0] as f64, pixel.0[1] as f64, pixel.0[2] as f64];

        let nearest = tree.nearest(&pxl, 1, &squared_euclidean)?;
        let nearest_tile = &tiles[*nearest[0].1];

        let x_offset_start = x * kernel_size.0;
        let y_offset_start = y * kernel_size.1;

        let mut x_offset_end = x_offset_start + kernel_size.0;
        let mut y_offset_end = y_offset_start + kernel_size.1;

        if x_offset_end > to_return.width() {
            x_offset_end = to_return.width();
        }
        if y_offset_end > to_return.height() {
            y_offset_end = to_return.height();
        }

        // copy tile to image
        for (ximage, xtile) in (x_offset_start..x_offset_end).zip(0..kernel_size.0) {
            for (yimage, ytile) in (y_offset_start..y_offset_end).zip(0..kernel_size.1) {
                let tile_pixel = unsafe { nearest_tile.unsafe_get_pixel(xtile, ytile) };
                unsafe { to_return.unsafe_put_pixel(ximage, yimage, tile_pixel) };
            }
        }
    }

    Ok(to_return)
}

pub fn build_mosaic_without_compression(
    src_image: &str,
    tiles: &[DynamicImage],
    kernel_size: (u32, u32),
) -> ResultSmall<DynamicImage> {
    let source = ImageReader::open(src_image)?.decode()?;

    let new_width = source.width() * kernel_size.0;
    let new_height = source.height() * kernel_size.1;

    let mut to_return = DynamicImage::new_rgb8(new_width, new_height);

    let mut tree: kdtree::KdTree<f64, usize, [f64; 3]> =
        kdtree::KdTree::with_capacity(3, tiles.len());

    for (index, tile) in tiles.iter().enumerate() {
        let color = mean_color(tile);

        tree.add(color, index)?;
    }

    for (x, y, pixel) in source.pixels() {
        let pxl = [pixel.0[0] as f64, pixel.0[1] as f64, pixel.0[2] as f64];

        let nearest = tree.nearest(&pxl, 1, &squared_euclidean)?;
        let nearest_tile = &tiles[*nearest[0].1];

        let x_offset_start = x * kernel_size.0;
        let y_offset_start = y * kernel_size.1;

        let mut x_offset_end = x_offset_start + kernel_size.0;
        let mut y_offset_end = y_offset_start + kernel_size.1;

        if x_offset_end > to_return.width() {
            x_offset_end = to_return.width();
        }
        if y_offset_end > to_return.height() {
            y_offset_end = to_return.height();
        }

        // copy tile to image
        for (ximage, xtile) in (x_offset_start..x_offset_end).zip(0..kernel_size.0) {
            for (yimage, ytile) in (y_offset_start..y_offset_end).zip(0..kernel_size.1) {
                let tile_pixel = unsafe{ nearest_tile.unsafe_get_pixel(xtile, ytile) };
                unsafe { to_return.unsafe_put_pixel(ximage, yimage, tile_pixel) };
            }
        }
    }

    Ok(to_return)
}


// /// opacity should be less than 255
// macro_rules! blend {
//     ($src:expr, $pxl:expr, $opacity:expr) => {
//         image::Rgba::<u8>([
//             $pxl.0[0]/(255-$opacity) + $src.0[0]/$opacity,
//             $pxl.0[1]/(255-$opacity) + $src.0[1]/$opacity,
//             $pxl.0[2]/(255-$opacity) + $src.0[2]/$opacity,
//             255
//         ])
//     };
// }

pub fn blend(
    src_pxl:image::Rgba<u8>, 
    pxl:image::Rgba<u8>, 
    opacity:u8
) -> image::Rgba<u8>{
    let pxl_opacity = (opacity as f64) / 255f64;
    let src_opacity = 1f64 - pxl_opacity;
    let res_opacity = 1f64 - (1f64 - pxl_opacity)*(1f64-src_opacity); 
    
    let r = ((pxl.0[0]as f64)/255f64)*pxl_opacity + ((src_pxl.0[0]as f64)/255f64)*src_opacity*(1f64-pxl_opacity);
    let g = ((pxl.0[1]as f64)/255f64)*pxl_opacity + ((src_pxl.0[1]as f64)/255f64)*src_opacity*(1f64-pxl_opacity);
    let b = ((pxl.0[2]as f64)/255f64)*pxl_opacity + ((src_pxl.0[2]as f64)/255f64)*src_opacity*(1f64-pxl_opacity);

    image::Rgba::<u8>([
        (r*255f64) as u8,
        (g*255f64) as u8,
        (b*255f64) as u8,
        (res_opacity*255f64) as u8
    ])

}

pub fn build_mosaic_blend(
    src_image: &str,
    tiles: &[DynamicImage],
    kernel_size: (u32, u32),
    opacity:u8,
) -> ResultSmall<DynamicImage> {

    let mut source = ImageReader::open(src_image)?.decode()?;

    let new_width = (source.width() as f64 / kernel_size.0 as f64).round() as u32;
    let new_height = (source.height() as f64 / kernel_size.1 as f64).round() as u32;

    let source_resized = source.resize_exact(new_width, new_height, FilterType::Nearest);

    let mut tree: kdtree::KdTree<f64, usize, [f64; 3]> = kdtree::KdTree::with_capacity(3, tiles.len());

    for (index, tile) in tiles.iter().enumerate() {
        let color = mean_color(tile);

        tree.add(color, index)?;
    }


    for (x, y, pixel) in source_resized.pixels() {
        let pxl = [pixel.0[0] as f64, pixel.0[1] as f64, pixel.0[2] as f64];

        let nearest = tree.nearest(&pxl, 1, &squared_euclidean)?;
        let nearest_tile = &tiles[*nearest[0].1];

        let x_offset_start = x * kernel_size.0;
        let y_offset_start = y * kernel_size.1;

        let mut x_offset_end = x_offset_start + kernel_size.0;
        let mut y_offset_end = y_offset_start + kernel_size.1;

        if x_offset_end > source.width() {
            x_offset_end = source.width();
        }
        if y_offset_end > source.height() {
            y_offset_end = source.height();
        }

        // copy tile to image
        for (ximage, xtile) in (x_offset_start..x_offset_end).zip(0..kernel_size.0) {
            for (yimage, ytile) in (y_offset_start..y_offset_end).zip(0..kernel_size.1) {
                let tile_pixel = unsafe { nearest_tile.unsafe_get_pixel(xtile, ytile) };
                let src_pixel = unsafe{ source.unsafe_get_pixel(ximage, yimage) };
                
                let new_pxl = blend(src_pixel,tile_pixel,opacity);
                unsafe { source.unsafe_put_pixel(ximage, yimage, new_pxl) };
            }
        }
    }


    Ok(source)
}
