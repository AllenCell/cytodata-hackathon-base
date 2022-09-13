import numpy as np


def rescale_image(img_data, channels=('bf', 'dna', 'membrane', 'structure',
                                      'dna_segmentation', 'membrane_segmentation', 'struct_segmentation_roof')):
    """
    'Raw' channels are stored with values between 0 and MAX_UINT16,
    where the 0-valued voxels denote the background. This function
    rescales the voxel values such that the background voxels become
    -1 and the remaining voxels become min-max scaled (between 0 and 1)
    """

    _MAX_UINT16 = 65535

    img_data = img_data.squeeze().astype(np.float32)

    for ix, channel in enumerate(channels):
        if "_seg" not in channel:
            img_data[ix] -= 1

            img_data[ix] = np.where(
                img_data[ix] >= 0,
                img_data[ix] / (_MAX_UINT16 - 1),
                -1
            )
    return img_data.astype(np.float16)
