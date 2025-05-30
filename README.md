# CardioSlice Viewer

An application for performing segmentation on cardiac magnetic resonance (CMR) volumes and visualizing the heart in 3D with synchronized 2D CMR slices.

Check out the [demo](https://youtu.be/O1wSYlpfjjc)!

![CardioSlice Viewer Demo Image](/images/Screenshot 2025-05-29 at 8.07.45PM.png)

---

## Features

- Load and visualize 3D CMR volumes (NIfTI)
- Run 3D UNet segmentation using sliding window inference
- Interactive 3D rendering via PyVista
- Axial, sagittal, and coronal views with adjustable sliders
- GUI built with PyQt5 and PyVistaQt

---

## Requirements

Install with pip:

```bash
pip install torch numpy nibabel pyvista scikit-image monai PyQt5 pyvistaqt matplotlib
```

---

## Additional Information

-- The 3D UNet model was trained on the HVSMR-2.0 [dataset](https://figshare.com/collections/HVSMR-2_0_A_3D_cardiovascular_MR_dataset_for_whole-heart_segmentation_in_congenital_heart_disease/7074755/2)
