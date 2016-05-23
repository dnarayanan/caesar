
try:
    import vtk
except:
    raise Exception('python-vtk could not be imported!\n' \
                    '           Install via `conda install vtk`')

import sys
import numpy as np

## keyboard shortcuts
shortcuts = dict(
    zoom_in = 'z',
    zoom_out = 'x',
    rotate_L = 'Left',
    rotate_R = 'Right',
    spin_L = 'Up',
    spin_R = 'Down',
    toggle_help = 't',
    save_file = 's',
    cam_settings = 'c',
    toggle_volume = 'v',
    toggle_actors = 'p'
)

class vtk_render(object):
    """Base class for the vtk wrapper."""

    def __init__(self):
        self.actors  = []
        self.volumes = []
        self.helpers = []

    def quit(self):
        sys.exit()

    def Keypress(self, obj, event):
        pass
        #key = obj.GetKeySym()

    def makebutton(self):
        pass
        #button = Button(text='quit', command=self.quit)
        #button.pack(expand='true', fill='x')

    def _set_input_connection(self,mapper,source):
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(source.GetOutput())
        else:
            mapper.SetInputConnection(source.GetOutputPort())

    def _set_input_data(self,mapper,source):
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(source)
        else:
            mapper.SetInputData(source)

    def point_render(self,pos,color=[1,1,1],opacity=1,alpha=1,psize=1):
        """Render a pointcloud in the scene.

        Parameters
        ----------
        pos: np.ndarray
            3D positions of points.
        color: list, np.ndarray, optional
            Color of points.  This can be a single RGB value
            or a list of RGB values (one per point).
        opacity: float, optional
            Transparency of the points.
        alpha: float, optional
            Transparency of the points (same as opacity).
        psize: int, optional
            Size of the points.

        """
        vtk_points = vtk.vtkPoints()

        if len(pos.shape) > 1:
            n_points = len(pos)
        elif pos.shape[-1] == 3:
            n_points = 1
            pos = np.vstack((pos,np.array([0,0,0])))
        else:
            return

        vtk_points.SetNumberOfPoints(n_points)
        for i in range(0,n_points):
            vtk_points.SetPoint(i,pos[i,0],pos[i,1],pos[i,2])

        vtk_cell_array = vtk.vtkCellArray()

        varying_colors = False
        if len(color) > 3:
            varying_colors = True

            Colors = vtk.vtkUnsignedCharArray()
            Colors.SetNumberOfComponents(3)
            Colors.SetName('Colors')

        for i in range(0,n_points):
            vtk_cell_array.InsertNextCell(1)
            vtk_cell_array.InsertCellPoint(i)

            if varying_colors:
                Colors.InsertNextTuple3(color[i,0]*255,
                                        color[i,1]*255,
                                        color[i,2]*255)

        vtk_poly_data = vtk.vtkPolyData()
        vtk_poly_data.SetPoints(vtk_points)
        vtk_poly_data.SetVerts(vtk_cell_array)

        if varying_colors:
            vtk_poly_data.GetPointData().SetScalars(Colors)
            vtk_poly_data.Modified()
            vtk_poly_data.Update()

        mapper = vtk.vtkPolyDataMapper()
        self._set_input_data(mapper,vtk_poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(psize)
        actor.GetProperty().SetOpacity(opacity)
        if not varying_colors:
            actor.GetProperty().SetColor(color[0],color[1],color[2])

        self.actors.append(actor)

    def draw_cube(self,center,size,color=[1,1,1]):
        """Draw a cube in the scene.

        Parameters
        ----------
        center: list, np.ndarray
            Center of the box in 3D space.
        size: float
            How large the box should be on a side.
        color: list, optional
            Color of the outline in RGB.

        """
        lxmin=center[0]-float(size)/2.
        lxmax=center[0]+float(size)/2.
        lymin=center[1]-float(size)/2.
        lymax=center[1]+float(size)/2.
        lzmin=center[2]-float(size)/2.
        lzmax=center[2]+float(size)/2.

        p1=[(lxmin,lymin,lzmin),(lxmin,lymin,lzmin),(lxmin,lymin,lzmin),
            (lxmin,lymax,lzmax),(lxmin,lymax,lzmax),(lxmin,lymax,lzmax),
            (lxmax,lymax,lzmax),(lxmax,lymax,lzmax),(lxmax,lymin,lzmax),
            (lxmax,lymin,lzmax),(lxmax,lymax,lzmin),(lxmax,lymax,lzmin)]
        p2=[(lxmax,lymin,lzmin),(lxmin,lymax,lzmin),(lxmin,lymin,lzmax),
            (lxmax,lymax,lzmax),(lxmin,lymin,lzmax),(lxmin,lymax,lzmin),
            (lxmax,lymax,lzmin),(lxmax,lymin,lzmax),(lxmin,lymin,lzmax),
            (lxmax,lymin,lzmin),(lxmin,lymax,lzmin),(lxmax,lymin,lzmin)]

        actors = []
        for i in range(12):
            source = vtk.vtkLineSource()
            source.SetPoint1(p1[i])
            source.SetPoint2(p2[i])
            mapper = vtk.vtkPolyDataMapper()
            self._set_input_connection(mapper,source)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actors.append(actor)
        self.actors.extend(actors)

    def draw_sphere(self,pos,r,color=[1,1,1],opacity=1,res=12):
        """Draw a sphere in the scene.

        Parameters
        ----------
        center: list, np.ndarray
            Center of the sphere in 3D space.
        r: float
            Radius of the sphere.
        color: list, optional
            Color of the sphere in RGB.
        opacity: float
            Transparency of the sphere.
        res: int
            Resolution of the sphere.

        """
        source = vtk.vtkSphereSource()
        source.SetCenter(pos[0],pos[1],pos[2])
        source.SetRadius(r)

        ## default res in phi & theta = 8
        source.SetPhiResolution(res)
        source.SetThetaResolution(res)

        mapper = vtk.vtkPolyDataMapper()
        self._set_input_connection(mapper,source)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetColor(color)

        self.actors.append(actor)

    def draw_arrow(self,p1,p2,shaft_r=0.01,tip_r=0.05,tip_l=0.2,
                   balls=0,ball_color=[1,1,1],ball_r=1,color=[1,1,1]):
        import random

        arrow_source = vtk.vtkArrowSource()

        random.seed(8775070)
        start_point = p1
        end_point   = p2

        # compute a basis
        normalized_x = [0,0,0]
        normalized_y = [0,0,0]
        normalized_z = [0,0,0]

        # the x axis is a vector from start to end
        math = vtk.vtkMath()
        math.Subtract(end_point, start_point, normalized_x)
        length = math.Norm(normalized_x)
        math.Normalize(normalized_x)

        # the z axis is an arbitrary vector cross x
        arb = [0,0,0]
        arb[0] = random.uniform(-10,10)
        arb[1] = random.uniform(-10,10)
        arb[2] = random.uniform(-10,10)
        math.Cross(normalized_x, arb, normalized_z)
        math.Normalize(normalized_z)

        # the y axis is z cross x
        math.Cross(normalized_z, normalized_x, normalized_y)
        matrix = vtk.vtkMatrix4x4()

        # create the direction cosin matrix
        matrix.Identity()
        for i in range(0,3):
            matrix.SetElement(i, 0, normalized_x[i])
            matrix.SetElement(i, 1, normalized_y[i])
            matrix.SetElement(i, 2, normalized_z[i])

        # apply the transforms
        transform = vtk.vtkTransform()
        transform.Translate(start_point)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # transform the poly data
        transform_pd = vtk.vtkTransformPolyDataFilter()
        transform_pd.SetTransform(transform)
        self._set_input_connection(transform_pd,arrow_source)

        mapper = []
        actors = []

        mapper.append(vtk.vtkPolyDataMapper())
        actors.append(vtk.vtkActor())

        self._set_input_connection(mapper[0],transform_pd)
        arrow_source.SetShaftRadius(shaft_r)  ## default is 0.03
        arrow_source.SetTipRadius(tip_r)      ## default is 0.1
        arrow_source.SetTipLength(tip_l)      ## default is 0.35
        actors[0].SetMapper(mapper[0])
        actors[0].GetProperty().SetColor(color)

        if balls:
            self.draw_sphere(start_point,ball_r,ball_color)
            self.draw_sphere(end_point,ball_r,ball_color)

        self.actors.extend(actors)


    def place_label(self,pos,text,text_color=[1,1,1],text_font_size=12,
                    label_box_color=[0,0,0],label_box=1,label_box_opacity=0.8):
        """Place a label in the scene.

        Parameters
        ----------
        pos : tuple, np.ndarray
            Position in 3D space where to place the label.
        text : str
            Label text.
        text_color : list, np.ndarray, optional
            Color of the label text in RGB.
        text_font_size : int, optional
            Text size of the label.
        label_box_color : list, np.ndarray, optional
            Background color of the label box in RGB.
        label_box: int, optional
            0=do not show the label box, 1=show the label box.
        label_box_opacity: float, optional
            Opacity value of the background box (1=no transparency).

        """
        pd     = vtk.vtkPolyData()
        pts    = vtk.vtkPoints()
        #verts  = vtk.vtkDoubleArray()
        orient = vtk.vtkDoubleArray()
        orient.SetName('orientation')
        label  = vtk.vtkStringArray()
        label.SetName('label')
        pts.InsertNextPoint(pos[0],pos[1],pos[2])
        orient.InsertNextValue(1)
        label.InsertNextValue(str(text))

        pd.SetPoints(pts)
        pd.GetPointData().AddArray(label)
        pd.GetPointData().AddArray(orient)


        h = vtk.vtkPointSetToLabelHierarchy()
        self._set_input_data(h,pd)
        h.SetOrientationArrayName('orientation')
        h.SetLabelArrayName('label')
        h.GetTextProperty().SetColor(text_color[0],text_color[1],text_color[2])
        h.GetTextProperty().SetFontSize(text_font_size)

        lmapper = vtk.vtkLabelPlacementMapper()
        self._set_input_connection(lmapper,h)
        if label_box:
            lmapper.SetShapeToRoundedRect()
        lmapper.SetBackgroundColor(label_box_color[0],
                                   label_box_color[1],
                                   label_box_color[2])
        lmapper.SetBackgroundOpacity(label_box_opacity)
        lmapper.SetMargin(3)

        lactor = vtk.vtkActor2D()
        lactor.SetMapper(lmapper)
        mapper = vtk.vtkPolyDataMapper()
        self._set_input_data(mapper,pd)
        actor  = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.actors.append(lactor)

    def _process_render(self,xsize,ysize):
        self.ren = vtk.vtkRenderer()
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.AddRenderer(self.ren)
        self.ren_win.SetSize(xsize,ysize)
        #self.ren_win.SetAAFrames(5)

        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.iren.SetRenderWindow(self.ren_win)

        for actor in self.actors:
            self.ren.AddActor(actor)

    def _setup_camera(self,focal_point):
        cam = self.ren.GetActiveCamera()

        if focal_point is not None:
            cam.SetFocalPoint(focal_point)

    def _orient_widget(self):
        self.axes = vtk.vtkAxesActor()
        self.axes.SetShaftTypeToCylinder()

        alt = vtk.vtkTextProperty()
        alt.SetColor([1,1,1])
        alt.SetFontSize(18)

        self.axes.GetXAxisCaptionActor2D().SetCaptionTextProperty(alt)
        self.axes.GetYAxisCaptionActor2D().SetCaptionTextProperty(alt)
        self.axes.GetZAxisCaptionActor2D().SetCaptionTextProperty(alt)

        self.marker = vtk.vtkOrientationMarkerWidget()
        self.marker.SetOrientationMarker(self.axes)
        self.marker.SetViewport(0,0,0.15,0.15)
        self.marker.SetInteractor(self.iren)
        self.marker.EnabledOn()
        self.marker.InteractiveOff()
        self.marker.KeyPressActivationOff()
        self.ren.ResetCamera()

    def render(self, xsize=800, ysize=800, bg_color=[0.5,0.5,0.5],
               focal_point=None, orient_widget=1):
        """Final call to render the window.

        Parameters
        ----------
        xsize : int, optional
            Horizontal size of the window in pixels.
        ysize : int, optional
            Vertical size of the window in pixels.
        bg_color : tuple, np.array, optional
            Background color in RGB.
        focal_point : tuple, np.array, optional
            Where to focus the camera on rendering.
        orient_widget : int, optional
            Show the orient widget?

        """

        self._process_render(xsize,ysize)
        self.ren.SetBackground(bg_color)

        self._orient_widget()
        self._setup_camera(focal_point)

        self.ren_win.Render()
        self.iren.Initialize()
        self.iren.AddObserver('KeyPressEvent',self.Keypress)
        self.iren.Start()
