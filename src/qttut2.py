# Qt application code

import typing
import requests
#from PyQt6 import QtCore
#from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from qgis.PyQt.QtWidgets import QAction, QMainWindow, QDockWidget, QVBoxLayout, QCheckBox, QSlider, QFrame, QFileDialog, QWidget, QPushButton, QLabel
from PyQt5.QtCore import Qt
from qgis.core import *
from qgis.gui import QgsMapCanvas, QgsLayerTreeView, QgsMapToolPan, QgsMapToolZoom
from PyQt5.QtGui import QIcon, QFont
import sys
#import processing
from processing.core.Processing import Processing, processing
class MyWnd(QMainWindow):
    def __init__(self, layer1,layer2,layer3,layer4):
        QMainWindow.__init__(self)

        self.canvas = QgsMapCanvas()
        self.canvas.setCanvasColor(Qt.white)
        self.setWindowTitle("Dataeaze BCD")
        self.setWindowIcon(QIcon("data/logo_dataeaze.jpeg"))
        self.setWindowIconText("datateaze")
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.canvas.setExtent(layer1.extent())
        self.canvas.setLayers([layer1,layer2,layer3,layer4])
        self.canvas.refresh()
        #iface.showAttributeTable(csv_layer)
        #layout = QDockWidget()
        
        checkbox3 = QCheckBox("Changes")
        checkbox3.setChecked(True) 
        checkbox2 = QCheckBox("P2")
        checkbox2.setChecked(True) 
        checkbox1 = QCheckBox("P1")
        checkbox1.setChecked(True)  # Set the initial state (checked)
        self.slider1 = QSlider()
        self.slider2 = QSlider()
        self.slider3 = QSlider()
        self.slider1_label3 = QLabel("Changes")
        self.slider1_label2 = QLabel("P2")
        self.slider1_label1 = QLabel("P1")
        self.slider1.setOrientation(1)  # Vertical orientation
        self.slider2.setOrientation(1)  # Vertical orientation
        self.slider3.setOrientation(1)  # Vertical orientation
        self.slider1.setRange(0, 100)  # Range from 0 to 100
        self.slider2.setRange(0, 100)  # Range from 0 to 100
        self.slider3.setRange(0, 100)  # Range from 0 to 100
        #self.layout1.addWidget(self.slider)
        self.apply_button1 = QPushButton("Apply")
        self.apply_button2 = QPushButton("Apply")
        self.apply_button3 = QPushButton("Apply")
        self.apply_button1.clicked.connect(self.apply_transparency1)
        self.apply_button2.clicked.connect(self.apply_transparency2)
        self.apply_button3.clicked.connect(self.apply_transparency3)
        
        self.widget1 = QWidget()
        self.layout1 = QVBoxLayout()
        self.widget1.setLayout(self.layout1)

        #self.layout1.addWidget(checkbox3)
        self.layout1.addWidget(self.slider1_label3)
        self.layout1.addWidget(self.slider3)
        self.layout1.addWidget(self.apply_button3)
        #self.layout1.addWidget(checkbox2)
        self.layout1.addWidget(self.slider1_label2)
        self.layout1.addWidget(self.slider2)
        self.layout1.addWidget(self.apply_button2)
        #self.layout1.addWidget(checkbox1)
        self.layout1.addWidget(self.slider1_label1)
        self.layout1.addWidget(self.slider1)
        self.layout1.addWidget(self.apply_button1)
        '''
        self.slider = QSlider()
        self.slider.setOrientation(1)  # Vertical orientation
        self.slider.setRange(0, 100)  # Range from 0 to 100
        self.layout1.addWidget(self.slider)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_transparency)
        self.layout1.addWidget(self.apply_button)'''

        #frame1 = QFrame()
        # Set the layout for the QFrame
        #frame1.setLayout(self.layout1)
        # # Set the size and position of the QFrame within the main window
        #frame1.setGeometry(500, 500, 500, 500)

        self.setCentralWidget(self.canvas)
        checkbox3.stateChanged.connect(self.checkbox_changed3)
        checkbox2.stateChanged.connect(self.checkbox_changed2)
        checkbox1.stateChanged.connect(self.checkbox_changed1)

        self.actionZoomIn = QAction("Zoom in", self)
        self.actionZoomOut = QAction("Zoom out", self)
        self.actionPan = QAction("Pan", self)

        self.actionZoomIn.setCheckable(True)
        self.actionZoomOut.setCheckable(True)
        self.actionPan.setCheckable(True)


        self.actionZoomIn.triggered.connect(self.zoomIn)
        self.actionZoomOut.triggered.connect(self.zoomOut)
        self.actionPan.triggered.connect(self.pan)


        self.toolbar = self.addToolBar("Canvas actions")

        self.toolbar.addAction(self.actionZoomIn)
        self.toolbar.addAction(self.actionZoomOut)
        self.toolbar.addAction(self.actionPan)
        
        # Section below relates to creating the 'layers panel'
        
        self.layers_widget = QDockWidget("Layers", self)
        self.layers_widget.setMaximumSize(100,300)
        self.layers_widget.setMinimumSize(100,300)
        self.layers_widget.move(10,50)  
        self.layers_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)  # To Fix the QDockWidget
        self.layers_widget.setWidget(self.widget1)
        '''
        self.view = QgsLayerTreeView(self.layers_widget)
        self.root = QgsProject.instance().layerTreeRoot()
        self.model = QgsLayerTreeModel(self.root)
        self.model.setFlag(QgsLayerTreeModel.AllowNodeReorder)
        self.model.setFlag(QgsLayerTreeModel.AllowNodeChangeVisibility)
        self.view.setModel(self.model)
        self.layers_widget.setWidget(self.view)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.layers_widget)'''

        # create the map tools
        self.toolPan = QgsMapToolPan(self.canvas)
        self.toolPan.setAction(self.actionPan)
        self.toolZoomIn = QgsMapToolZoom(self.canvas, False) # false = in
        self.toolZoomIn.setAction(self.actionZoomIn)
        self.toolZoomOut = QgsMapToolZoom(self.canvas, True) # true = out
        self.toolZoomOut.setAction(self.actionZoomOut)

        self.pan()

    

    def zoomIn(self):
        self.canvas.setMapTool(self.toolZoomIn)

    def zoomOut(self):
        self.canvas.setMapTool(self.toolZoomOut)

    def pan(self):
        self.canvas.setMapTool(self.toolPan)
        
            # Connect the checkbox to a function
    def checkbox_changed1(self,state):
        if state == 2:  # 2 represents checked state
            #self.layer1.setOpacity(0.7)
            self.canvas.setLayers([self.layer1])
        else:
            self.canvas.setLayers([])

    def checkbox_changed2(self,state):
        if state == 2:  # 2 represents checked state
            self.canvas.setLayers([self.layer2])
        else:
            self.canvas.setLayers([])
    
    def checkbox_changed3(self,state):
        if state == 2:  # 2 represents checked state
            self.canvas.setLayers([self.layer3])
        else:
            self.canvas.setLayers([])

    def apply_transparency1(self):
        transparency_value = self.slider1.value() / 100.0
        self.layer1.setOpacity(transparency_value)
        self.layer1.triggerRepaint()

    def apply_transparency2(self):
        transparency_value = self.slider2.value() / 100.0
        self.layer2.setOpacity(transparency_value)
        self.layer2.triggerRepaint()
    
    def apply_transparency3(self):
        transparency_value = self.slider3.value() / 100.0
        self.layer3.setOpacity(transparency_value)
        self.layer3.triggerRepaint()



#QgsApplication.setPrefixPath("/home/user/Desktop/ind/",True )
qgs = QgsApplication([],True)

QgsApplication.initQgis()
Processing.initialize()
x = QgsProject.instance()
path_to_tif = "data/p1ntif.tif"
file_dialog1 = QFileDialog()
path_to_tif, _ = file_dialog1.getOpenFileName(None, "Select past image", "", "All Files (*)")
print(path_to_tif)

path_to_tif2 = "data/p2nmsetif.tif"
file_dialog2 = QFileDialog()
path_to_tif2, _ = file_dialog2.getOpenFileName(None, "Select latest image", "", "All Files (*)")
print(path_to_tif2)
p1ntif = QgsRasterLayer(path_to_tif, "p1ntif")


'''
# This approch is not working
new_crs = QgsCoordinateReferenceSystem('EPSG:32643')
p1ntif.setCrs(new_crs)
# Update the layer's extent to match the new CRS
p1ntif.triggerRepaint()
print('Current CRS after change:', p1ntif.crs().authid(), p1ntif.crs().description())'''



# Check if the raster layer was loaded successfully
if not p1ntif.isValid():
    print('Raster layer failed to load!')
else:
    # Create a QgsCoordinateReferenceSystem object with the desired CRS
    new_crs = QgsCoordinateReferenceSystem('EPSG:3857')  # Change to your desired CRS

    # Create a processing context and feedback object
    context = QgsProcessingContext()
    feedback = QgsProcessingFeedback()

    # Define the output path for the reprojected raster
    projected_raster_path = 'data/newprojected_p1ntif.tif'

    # Reproject the raster layer
    parameters = {
        'INPUT': p1ntif,
        'TARGET_CRS': new_crs,
        'OUTPUT': projected_raster_path,
    }
    
    processing_algorithm = 'gdal:warpreproject'
    feedback.pushInfo(f"Reprojecting to {new_crs.authid()}")

    result = processing.run(processing_algorithm, parameters, context=context, feedback=feedback)

    # Check if the reprojected raster was created successfully
    if result['OUTPUT']:
        # Load the projected raster layer
        projected_raster_layer1 = QgsRasterLayer(projected_raster_path, 'Projected Raster', 'gdal')

        # Check if the projected raster layer was loaded successfully
        if not projected_raster_layer1.isValid():
            print('Projected raster layer failed to load!')
        else:
            # Add the projected raster layer to the map
            print("projected layer is ready")
            #x.addMapLayer(projected_raster_layer)
    else:
        print('Reprojection failed.')



print("--------->",p1ntif.isValid())
p2ntif = QgsRasterLayer(path_to_tif2, "p2ntif")


if not p2ntif.isValid():
    print('Raster layer failed to load!')
else:
    # Create a QgsCoordinateReferenceSystem object with the desired CRS
    new_crs = QgsCoordinateReferenceSystem('EPSG:3857')  # Change to your desired CRS

    # Create a processing context and feedback object
    context = QgsProcessingContext()
    feedback = QgsProcessingFeedback()

    # Define the output path for the reprojected raster
    projected_raster_path = 'data/newprojected_p2ntif.tif'

    # Reproject the raster layer
    parameters = {
        'INPUT': p2ntif,
        'TARGET_CRS': new_crs,
        'OUTPUT': projected_raster_path,
    }
    
    processing_algorithm = 'gdal:warpreproject'
    feedback.pushInfo(f"Reprojecting to {new_crs.authid()}")

    result = processing.run(processing_algorithm, parameters, context=context, feedback=feedback)

    # Check if the reprojected raster was created successfully
    if result['OUTPUT']:
        # Load the projected raster layer
        projected_raster_layer2 = QgsRasterLayer(projected_raster_path, 'Projected Raster', 'gdal')

        # Check if the projected raster layer was loaded successfully
        if not projected_raster_layer2.isValid():
            print('Projected raster layer failed to load!')
        else:
            # Add the projected raster layer to the map
            print("projected layer is ready")
            #x.addMapLayer(projected_raster_layer)
    else:
        print('Reprojection failed.')


print("--------->",p2ntif.isValid())

path_to_tif_res = "data/georef_newdataset1_filtered900.tif"
restif = QgsRasterLayer(path_to_tif_res, "restif")

if not restif.isValid():
    print('Raster layer failed to load!')
else:
    # Create a QgsCoordinateReferenceSystem object with the desired CRS
    new_crs = QgsCoordinateReferenceSystem('EPSG:3857')  # Change to your desired CRS

    # Create a processing context and feedback object
    context = QgsProcessingContext()
    feedback = QgsProcessingFeedback()

    # Define the output path for the reprojected raster
    projected_raster_path = 'data/newprojected_result.tif'

    # Reproject the raster layer
    parameters = {
        'INPUT': restif,
        'TARGET_CRS': new_crs,
        'OUTPUT': projected_raster_path,
    }
    
    processing_algorithm = 'gdal:warpreproject'
    feedback.pushInfo(f"Reprojecting to {new_crs.authid()}")

    result = processing.run(processing_algorithm, parameters, context=context, feedback=feedback)

    # Check if the reprojected raster was created successfully
    if result['OUTPUT']:
        # Load the projected raster layer
        projected_raster_layer3 = QgsRasterLayer(projected_raster_path, 'Projected Raster', 'gdal')

        # Check if the projected raster layer was loaded successfully
        if not projected_raster_layer3.isValid():
            print('Projected raster layer failed to load!')
        else:
            # Add the projected raster layer to the map
            print("projected layer is ready")
            #x.addMapLayer(projected_raster_layer)
    else:
        print('Reprojection failed.')

print("--------->",restif.isValid())
print(QgsApplication.prefixPath())

#qgsProject = QgsProject.instance()
channel_path = "mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
channel_path2 = "http://tile.openstreetmap.org/{z}/{x}/{y}.png"
service_uri = "type=xyz&url=https://"+requests.utils.quote(channel_path2)
service_uri = "type=xyz&url=http://tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png"
channel_layer = QgsRasterLayer(service_uri, 'channel_layer', 'wms')  # EDIT THIS LINE
if channel_layer.isValid():
    print("channel_layer loaded")
else:
    print("Channel layer is not added")

#csv_file_path = "/home/user/Desktop/ind/scripts/sample3.csv"
#csv_layer = QgsVectorLayer(csv_file_path, "My CSV Layer", "ogr")
#if csv_layer.isValid():
    # Open the attribute table for the CSV layer
    #QgsVectorLayerUtils.openTable(csv_layer)
#attribute_table = QgsVectorLayer.getAvailableAttributeFields(csv_layer)
#csv_layer.openAttributeTable(attribute_table)

w = MyWnd(x.addMapLayer(projected_raster_layer1),x.addMapLayer(projected_raster_layer2),x.addMapLayer(projected_raster_layer3),x.addMapLayer(channel_layer))
w.show()
#sys.exit(app.exec_())
sys.exit(qgs.exec_())

