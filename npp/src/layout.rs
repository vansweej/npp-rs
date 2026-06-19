#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Describes the memory layout of a packed NPP-compatible image buffer.
pub struct CudaLayout {
    /// The number of channels in the color representation of the image.
    pub channels: u8,

    /// Add this to an index to get to the sample in the next channel.
    pub channel_stride: usize,

    /// The width of the represented image.
    pub width: u32,

    /// Add this to an index to get to the next sample in x-direction.
    pub width_stride: usize,

    /// The height of the represented image.
    pub height: u32,

    /// Add this to an index to get to the next sample in y-direction.
    pub height_stride: usize,

    /// index of the left upper pixel of the image, this property is used with
    /// sub images in existing images. When this layout describes a full picture
    /// it is 0.
    pub img_index: usize,
}

impl CudaLayout {
    /// Describe a row-major image packed in all directions.
    ///
    /// # Panics
    ///
    /// On platforms where `usize` has the same size as `u32` this panics when the resulting stride
    /// in the `height` direction would be larger than `usize::max_value()`. On other platforms
    /// where it can surely accomodate `u8::max_value() * u32::max_value(), this can never happen.
    pub fn row_major_packed(channels: u8, width: u32, height: u32) -> Self {
        let height_stride = (channels as usize).checked_mul(width as usize).expect(
            "Row major packed image can not be described because it does not fit into memory",
        );
        CudaLayout {
            channels,
            channel_stride: 1,
            width,
            width_stride: channels as usize,
            height,
            height_stride,
            img_index: 0,
        }
    }

    /// Get the image dimensions as (channels, width, height).
    pub fn dimensions(&self) -> (u8, u32, u32) {
        (self.channels, self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_major_packed() {
        let layout = CudaLayout::row_major_packed(3, 640, 480);
        assert_eq!(layout.channels, 3);
        assert_eq!(layout.channel_stride, 1);
        assert_eq!(layout.width, 640);
        assert_eq!(layout.width_stride, 3);
        assert_eq!(layout.height, 480);
        assert_eq!(layout.height_stride, 1920);
        assert_eq!(layout.img_index, 0);
    }
}
