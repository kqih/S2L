import cv2
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt


CUPY_AVAILABLE = False

class ROIVisualizer:
    def __init__(self, label_image_path, original_image_path, excel_output_path, plot_output_path, show_labels=False, progress_callback=None):
        self.label_image_path = label_image_path
        self.original_image_path = original_image_path
        self.excel_output_path = excel_output_path
        self.plot_output_path = plot_output_path
        self.label_rois = show_labels
        self.zoom_factor = 1.0
        self.zoom_center = (0.5, 0.5)
        self.progress_callback = progress_callback
        self.setup_plot()

    def setup_plot(self):
        """Set up the Matplotlib figure."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(left=0.1, bottom=0.1)
        self.update_plot()

    def create_overlays(self):
        """Create overlay images with and without labels."""
        label_image = cv2.imread(self.label_image_path, cv2.IMREAD_UNCHANGED)
        original_image = cv2.imread(self.original_image_path, cv2.IMREAD_COLOR)

        if label_image is None or original_image is None:
            raise ValueError("One or both images could not be loaded. Check the file paths.")

        if CUPY_AVAILABLE:
            label_image_cp = cp.asarray(label_image)
            original_image_cp = cp.asarray(original_image)

            unique_labels = cp.unique(label_image_cp)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background

            colors = cp.random.randint(0, 256, (len(unique_labels), 3), dtype=cp.uint8)

            overlay_image_cp = cp.zeros_like(original_image_cp)
            label_overlay_cp = cp.zeros_like(original_image_cp)

            for idx, label in enumerate(unique_labels.get()):
                mask = (label_image_cp == label)
                overlay_image_cp[mask] = colors[idx]
                label_overlay_cp[mask] = colors[idx]

            # Convert label_overlay_cp to NumPy once after all operations
            label_overlay_np = cp.asnumpy(label_overlay_cp)
            for idx, label in enumerate(unique_labels.get()):
                mask = (label_image_cp == label).get()
                coords = np.nonzero(mask)
                if coords[0].size > 0:
                    center_y, center_x = int(np.mean(coords[0])), int(np.mean(coords[1]))
                    cv2.putText(label_overlay_np, str(label), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            self.roi_image = cp.asnumpy(overlay_image_cp)
            self.label_roi_image = label_overlay_np
        else:
            unique_labels = np.unique(label_image)
            unique_labels = unique_labels[unique_labels > 0]

            colors = np.random.randint(0, 256, (len(unique_labels), 3), dtype=np.uint8)

            overlay_image_np = np.zeros_like(original_image)
            label_overlay_np = np.zeros_like(original_image)

            for idx, label in enumerate(unique_labels):
                mask = (label_image == label)
                overlay_image_np[mask] = colors[idx]
                label_overlay_np[mask] = colors[idx]

            # Add text labels in a single pass
            for idx, label in enumerate(unique_labels):
                mask = (label_image == label)
                coords = np.nonzero(mask)
                if coords[0].size > 0:
                    center_y, center_x = int(np.mean(coords[0])), int(np.mean(coords[1]))
                    cv2.putText(label_overlay_np, str(label), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            self.roi_image = overlay_image_np
            self.label_roi_image = label_overlay_np

        alpha = 0.2
        self.combined_image = cv2.addWeighted(original_image, alpha, self.label_roi_image, 1 - alpha, 0)

    def update_plot(self):
        """Update the plot with or without ROI labels based on the label_rois attribute."""
        # Create overlays if they haven't been created yet
        if not hasattr(self, 'roi_image'):
            self.create_overlays()

        image_to_show = self.combined_image if self.label_rois else self.roi_image

        self.ax.clear()

        h, w, _ = image_to_show.shape
        cx, cy = int(w * self.zoom_center[0]), int(h * self.zoom_center[1])
        zoom_w, zoom_h = int(w / self.zoom_factor), int(h / self.zoom_factor)
        zoom_x1, zoom_x2 = max(0, cx - zoom_w // 2), min(w, cx + zoom_w // 2)
        zoom_y1, zoom_y2 = max(0, cy - zoom_h // 2), min(h, cy + zoom_h // 2)
        zoom_image = image_to_show[zoom_y1:zoom_y2, zoom_x1:zoom_x2]

        self.ax.imshow(cv2.cvtColor(zoom_image, cv2.COLOR_BGR2RGB))
        self.ax.set_title('ROIs Overlay')
        self.ax.axis('off')

        self.fig.savefig(self.plot_output_path, dpi=300)

    def save_rois_to_excel(self):
        """Save ROI information to an Excel spreadsheet."""
        label_image = cv2.imread(self.label_image_path, cv2.IMREAD_UNCHANGED)
        original_image = cv2.imread(self.original_image_path, cv2.IMREAD_GRAYSCALE)

        if label_image is None or original_image is None:
            raise ValueError("Unable to load image(s) at provided path(s).")

        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels > 0]

        roi_data = []

        for label in unique_labels:
            mask = label_image == label
            area = np.sum(mask)
            integrated_density = np.sum(original_image[mask])
            mean_value = np.mean(original_image[mask])
            std_dev = np.std(original_image[mask])
            roi_data.append([label, area, integrated_density, mean_value, std_dev])

        # Save to Excel using xlsxwriter
        with xlsxwriter.Workbook(self.excel_output_path) as workbook:
            worksheet = workbook.add_worksheet()

            headers = ['Label', 'Area', 'Integrated Density', 'Mean Gray Value', 'Standard Deviation']
            worksheet.write_row(0, 0, headers)

            for row, data in enumerate(roi_data, start=1):
                worksheet.write_row(row, 0, data)

        print(f"ROI information saved to {self.excel_output_path}")

        # Call progress callback with 100% completion after saving
        if self.progress_callback:
            self.progress_callback(100)
