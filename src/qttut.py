#!/usr/bin/env python3
"""
QGIS + PyQt Application for Binary Change Detection Visualization
"""

import sys
import argparse
import requests
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QVBoxLayout, QCheckBox, QSlider, QWidget, QPushButton, QLabel, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from qgis.core import (
    QgsApplication, QgsRasterLayer, QgsCoordinateReferenceSystem,
    QgsProcessingContext, QgsProcessingFeedback, QgsProject
)
from qgis.gui import QgsMapCanvas, QgsMapToolPan, QgsMapToolZoom
from processing.core.Processing import Processing, processing


class MyWnd(QMainWindow):
    """Main QGIS Map Window with layer transparency controls."""

    def __init__(self, layer1, layer2, layer3, layer4):
        super().__init__()

        self.setWindowTitle("Dataeaze BCD")
        self.setWindowIcon(QIcon("data/logo_dataeaze.jpeg"))

        # Map canvas
        self.canvas = QgsMapCanvas()
        self.canvas.setCanvasColor(Qt.white)
        self.canvas.setExtent(layer1.extent())
        self.canvas.setLayers([layer1, layer2, layer3, layer4])
        self.setCentralWidget(self.canvas)

        # Layer references
        self.layer1, self.layer2, self.layer3 = layer1, layer2, layer3

        # Layer controls dock
        self.layers_widget = QDockWidget("Layers", self)
        self.layers_widget.setMaximumSize(150, 300)
        self.layers_widget.setMinimumSize(150, 300)
        self.layers_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.slider1, self.slider2, self.slider3 = QSlider(), QSlider(), QSlider()
        for s in (self.slider1, self.slider2, self.slider3):
            s.setOrientation(Qt.Vertical)
            s.setRange(0, 100)

        self.apply_button1 = QPushButton("Apply")
        self.apply_button2 = QPushButton("Apply")
        self.apply_button3 = QPushButton("Apply")

        self.apply_button1.clicked.connect(lambda: self.apply_transparency(self.layer1, self.slider1))
        self.apply_button2.clicked.connect(lambda: self.apply_transparency(self.layer2, self.slider2))
        self.apply_button3.clicked.connect(lambda: self.apply_transparency(self.layer3, self.slider3))

        layout = QVBoxLayout()
        for lbl, sld, btn, name in [
            (QLabel("Changes"), self.slider3, self.apply_button3, "Changes"),
            (QLabel("P2"), self.slider2, self.apply_button2, "P2"),
            (QLabel("P1"), self.slider1, self.apply_button1, "P1"),
        ]:
            layout.addWidget(lbl)
            layout.addWidget(sld)
            layout.addWidget(btn)

        widget = QWidget()
        widget.setLayout(layout)
        self.layers_widget.setWidget(widget)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.layers_widget)

        # Toolbar actions
        self.actionZoomIn = self.addAction("Zoom in", self.zoomIn)
        self.actionZoomOut = self.addAction("Zoom out", self.zoomOut)
        self.actionPan = self.addAction("Pan", self.pan)

        for action in (self.actionZoomIn, self.actionZoomOut, self.actionPan):
            action.setCheckable(True)

        self.toolbar = self.addToolBar("Canvas actions")
        self.toolbar.addAction(self.actionZoomIn)
        self.toolbar.addAction(self.actionZoomOut)
        self.toolbar.addAction(self.actionPan)

        # Map tools
        self.toolPan = QgsMapToolPan(self.canvas)
        self.toolPan.setAction(self.actionPan)
        self.toolZoomIn = QgsMapToolZoom(self.canvas, False)
        self.toolZoomIn.setAction(self.actionZoomIn)
        self.toolZoomOut = QgsMapToolZoom(self.canvas, True)
        self.toolZoomOut.setAction(self.actionZoomOut)

        self.pan()

    def zoomIn(self):
        self.canvas.setMapTool(self.toolZoomIn)

    def zoomOut(self):
        self.canvas.setMapTool(self.toolZoomOut)

    def pan(self):
        self.canvas.setMapTool(self.toolPan)

    def apply_transparency(self, layer, slider):
        value = slider.value() / 100.0
        layer.setOpacity(value)
        layer.triggerRepaint()


def reproject_raster(layer_path, out_path, crs="EPSG:3857"):
    """Reproject raster to target CRS using GDAL warp via QGIS processing."""
    layer = QgsRasterLayer(layer_path, layer_path.split("/")[-1])
    if not layer.isValid():
        raise RuntimeError(f"Raster {layer_path} failed to load!")

    new_crs = QgsCoordinateReferenceSystem(crs)
    context = QgsProcessingContext()
    feedback = QgsProcessingFeedback()

    parameters = {
        'INPUT': layer,
        'TARGET_CRS': new_crs,
        'OUTPUT': out_path,
    }
    result = processing.run('gdal:warpreproject', parameters, context=context, feedback=feedback)
    return QgsRasterLayer(result['OUTPUT'], f"Reprojected_{layer.name()}", "gdal")


def load_basemap():
    """Load OpenStreetMap XYZ layer."""
    service_uri = "type=xyz&url=http://tile.openstreetmap.org/{z}/{x}/{y}.png"
    layer = QgsRasterLayer(service_uri, 'OSM Basemap', 'wms')
    if not layer.isValid():
        raise RuntimeError("Basemap failed to load")
    return layer


def main():
    parser = argparse.ArgumentParser(description="QGIS + PyQt GUI for Binary Change Detection")
    parser.add_argument("--p1", required=True, help="Path to past image (TIF)")
    parser.add_argument("--p2", required=True, help="Path to latest image (TIF)")
    parser.add_argument("--res", required=True, help="Path to result/changes raster (TIF)")
    args = parser.parse_args()

    QgsApplication.setPrefixPath("/usr", True)
    qgs = QgsApplication([], True)
    QgsApplication.initQgis()
    Processing.initialize()

    # Reproject layers
    p1_layer = reproject_raster(args.p1, "data/proj_p1.tif")
    p2_layer = reproject_raster(args.p2, "data/proj_p2.tif")
    res_layer = reproject_raster(args.res, "data/proj_res.tif")
    base_layer = load_basemap()

    # Launch window
    win = MyWnd(p1_layer, p2_layer, res_layer, base_layer)
    win.show()

    sys.exit(qgs.exec_())


if __name__ == "__main__":
    main()

