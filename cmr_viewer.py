import sys
import torch
import numpy as np
import nibabel as nib
import pyvista as pv
from skimage import measure
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Spacing, Orientation,
    ScaleIntensityRange, CropForeground, ResizeWithPadOrCrop,
    EnsureType, ToTensor
)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QMessageBox, QSlider, QLabel
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from pyvistaqt import QtInteractor
import matplotlib.pyplot as plt


class SegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CardioSlice Viewer")
        self.resize(1200, 700)

        self.volume_data = None
        self.segmentation = None
        self.spacing = None
        self.orientation = 'axial'

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.btn_load = QPushButton("Load NIfTI Image and Segment")
        self.btn_load.clicked.connect(self.load_and_segment)
        main_layout.addWidget(self.btn_load)

        h_layout = QHBoxLayout()
        main_layout.addLayout(h_layout)

        self.plotter = QtInteractor(self)
        h_layout.addWidget(self.plotter.interactor, stretch=2)

        right_layout = QVBoxLayout()
        h_layout.addLayout(right_layout, stretch=1)

        self.slice_label = QLabel("Slice View")
        self.slice_label.setAlignment(Qt.AlignCenter)
        self.slice_label.setFixedSize(300, 300)
        right_layout.addWidget(self.slice_label)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setTickPosition(QSlider.TicksBelow)
        self.slice_slider.setTickInterval(1)
        self.slice_slider.valueChanged.connect(lambda val: self.update_slice_plane(val, orientation=self.orientation))
        self.slice_slider.setEnabled(False)
        right_layout.addWidget(self.slice_slider)

        orientation_layout = QHBoxLayout()
        self.axial_btn = QPushButton("Axial")
        self.axial_btn.clicked.connect(lambda: self.set_orientation("axial"))
        orientation_layout.addWidget(self.axial_btn)

        self.sagittal_btn = QPushButton("Sagittal")
        self.sagittal_btn.clicked.connect(lambda: self.set_orientation("sagittal"))
        orientation_layout.addWidget(self.sagittal_btn)

        self.coronal_btn = QPushButton("Coronal")
        self.coronal_btn.clicked.connect(lambda: self.set_orientation("coronal"))
        orientation_layout.addWidget(self.coronal_btn)

        right_layout.addLayout(orientation_layout)

        self.device = torch.device("cpu")
        self.model_path = "/Users/matthewkim/Documents/cardioslice/final_unet_model_epoch_100.pth"
        self.img_size = (128, 128, 128)
        self.num_classes = 9
        self.model = self.load_model()

        self.slice_plane_mesh = None

    def load_model(self):
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=self.num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        return model

    def preprocess(self, image_path):
        transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Spacing(pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(a_min=-100, a_max=500, b_min=0.0, b_max=1.0, clip=True),
            CropForeground(source_key=None),
            ResizeWithPadOrCrop(spatial_size=self.img_size),
            EnsureType(),
            ToTensor()
        ])
        img_tensor = transforms(image_path)
        return img_tensor.unsqueeze(0)

    def set_orientation(self, orientation):
        self.orientation = orientation
        if self.volume_data is None:
            return

        if orientation == 'axial':
            max_index = self.volume_data.shape[1] - 1
        elif orientation == 'sagittal':
            max_index = self.volume_data.shape[2] - 1
        elif orientation == 'coronal':
            max_index = self.volume_data.shape[0] - 1
        else:
            return

        self.slice_slider.blockSignals(True)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(max_index)
        self.slice_slider.setValue(max_index // 2)
        self.slice_slider.blockSignals(False)

        self.update_slice_plane(self.slice_slider.value(), orientation)

    def visualize_multiclass_segmentation(self, pred_label, spacing):
        self.plotter.clear()
        colors = {
            0: [0, 0, 0, 0],
            1: [1, 0, 0, 0.4],
            2: [0, 1, 0, 0.4],
            3: [0, 0, 1, 0.4],
            4: [0.0, 0.5, 0.5, 0.4],
            5: [1, 0, 1, 0.4],
            6: [0, 1, 1, 0.4],
            7: [1, 0.5, 0, 0.4],
            8: [0.5, 0, 0.5, 0.4],
        }

        label_names = {
            1: "Left Ventricle",
            2: "Right Ventricle",
            3: "Left Atrium",
            4: "Right Atrium",
            5: "Ascending Aorta",
            6: "Pulmonary Artery",
            7: "Superior Vena Cava",
            8: "Inferior Vena Cava",
        }

        for label in range(1, 9):
            mask = (pred_label == label).astype(np.uint8)
            if np.sum(mask) == 0:
                continue
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=0)
            except ValueError:
                continue
            
            verts = verts * spacing
            faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
            mesh = pv.PolyData(verts, faces)
            color_rgb = [int(255 * c) for c in colors[label][:3]]
            self.plotter.add_mesh(mesh, color=color_rgb, opacity=colors[label][3], name=f"Label_{label}")

        legend = [(label_names[i], [int(255 * c) for c in colors[i][:3]]) for i in range(1, 9)]
        self.plotter.add_legend(legend, bcolor='w', border=True)
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self.plotter.render()

    def voxel_to_world(self, affine, i, j, k):
        voxel_coord = np.array([i, j, k, 1])
        world_coord = affine @ voxel_coord
        return world_coord[:3]

    def update_slice_plane(self, slice_index, orientation='axial'):
        if self.volume_data is None:
            return

        affine = self.affine
        vol = self.volume_data

        if orientation == 'axial':
            i_center = (vol.shape[0] - 1) / 2
            j = slice_index
            k_center = (vol.shape[2] - 1) / 2
            
            center = self.voxel_to_world(affine, i_center, j, k_center)

            p1 = self.voxel_to_world(affine, 0, j, 0)
            p2 = self.voxel_to_world(affine, vol.shape[0] - 1, j, 0)
            p3 = self.voxel_to_world(affine, 0, j, vol.shape[2] - 1)
            # Note: recentering was adjusted based on viewing the associated 2D slice with
            # the 3D segmentation with the example file. This does not work ideally for all
            # inputs. The logic for synchronization is still being adjusted.
            center[0] = p3[0]
            center[1] = p3[1]

            i_size = np.linalg.norm(p3 - p1)
            j_size = np.linalg.norm(p2 - p1)

            normal = affine[:3, 1]
            normal = normal / np.linalg.norm(normal)

            plane = pv.Plane(center=center, direction=normal, i_size=i_size, j_size=j_size)

            # Orienting the slice to match with 3D segmentation
            slice_img_3d = vol[:, slice_index, :].T
            slice_img_3d = np.rot90(slice_img_3d, k=3)
            slice_img_side = vol[:, slice_index, :]
            slice_img_side = np.flipud(slice_img_side)
            slice_img_side = np.fliplr(slice_img_side)
        elif orientation == 'sagittal':
            i_center = vol.shape[0] / 2
            j_center = vol.shape[1] / 2
            k = slice_index
            center = self.voxel_to_world(affine, i_center, j_center, k)

            p1 = self.voxel_to_world(affine, 0, 0, k)
            p2 = self.voxel_to_world(affine, vol.shape[0], 0, k)
            p3 = self.voxel_to_world(affine, 0, vol.shape[1], k)
            center[0] += p3[1]
            center[1] = p3[1]

            i_size = np.linalg.norm(p3 - p1)
            j_size = np.linalg.norm(p2 - p1)

            normal = affine[:3, 2]
            normal = normal / np.linalg.norm(normal)

            plane = pv.Plane(center=center, direction=normal, i_size=i_size, j_size=j_size)

            slice_img_3d = vol[:, :, slice_index].T
            slice_img_3d = np.rot90(slice_img_3d)
            slice_img_3d = np.flipud(slice_img_3d)
            slice_img_side = vol[:, :, slice_index].T
        elif orientation == 'coronal':
            i = slice_index
            j_center = vol.shape[1] / 2
            k_center = vol.shape[2] / 2
            center = self.voxel_to_world(affine, i, j_center, k_center)

            p1 = self.voxel_to_world(affine, i, 0, 0)
            p2 = self.voxel_to_world(affine, i, vol.shape[1], 0)
            p3 = self.voxel_to_world(affine, i, 0, vol.shape[2])
            center[0] = p3[0]
            center[1] += p3[0]

            i_size = np.linalg.norm(p2 - p1)
            j_size = np.linalg.norm(p3 - p1)

            normal = affine[:3, 0]
            normal = normal / np.linalg.norm(normal)

            plane = pv.Plane(center=center, direction=normal, i_size=i_size, j_size=j_size)

            slice_img_3d = vol[slice_index, :, :].T
            slice_img_3d = np.flipud(slice_img_3d)
            slice_img_side = vol[slice_index, :, :]
            slice_img_side = np.fliplr(slice_img_side)
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

        slice_img_norm_3d = ((slice_img_3d - slice_img_3d.min()) / (np.ptp(slice_img_3d) + 1e-8) * 255).astype(np.uint8)
        texture = pv.numpy_to_texture(slice_img_norm_3d)

        if self.slice_plane_mesh:
            self.plotter.remove_actor(self.slice_plane_mesh)

        self.slice_plane_mesh = self.plotter.add_mesh(plane, texture=texture, name="Slice_Plane")
        self.show_slice_in_label(((slice_img_side - slice_img_side.min()) / (np.ptp(slice_img_side) + 1e-8) * 255).astype(np.uint8))
        self.plotter.render()

    def show_slice_in_label(self, slice_img):
        h, w = slice_img.shape
        slice_bytes = slice_img.astype(np.uint8).tobytes()
        qimage = QImage(slice_bytes, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.slice_label.size(), Qt.KeepAspectRatio)
        self.slice_label.setPixmap(pixmap)

    def load_and_segment(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a NIfTI File", "", "NIfTI files (*.nii *.nii.gz);;All Files (*)")
        if not file_path:
            QMessageBox.warning(self, "No file selected", "Please select a NIfTI file to proceed.")
            return

        try:
            nii_img = nib.load(file_path)
            self.affine = nii_img.affine
            spacing = nii_img.header.get_zooms()[:3]
            volume = nii_img.get_fdata()
        except Exception as e:
            QMessageBox.critical(self, "File Error", f"Failed to load NIfTI file:\n{e}")
            return

        self.volume_data = volume.astype(np.float32)
        self.spacing = spacing

        img_tensor = self.preprocess(file_path).to(self.device)

        with torch.no_grad():
            output = sliding_window_inference(img_tensor, roi_size=self.img_size, sw_batch_size=1, predictor=self.model)
            pred_label = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        self.segmentation = pred_label

        self.visualize_multiclass_segmentation(pred_label, spacing)

        self.slice_slider.setEnabled(True)
        self.set_orientation("axial")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SegmentationApp()
    window.show()
    sys.exit(app.exec_())