from kivy.config import Config 

Config.set('graphics', 'resizable', '1') 
Config.set('graphics', 'width', '1920') 
Config.set('graphics', 'height', '1080')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

import kivy
kivy.require('1.10.1') # replace with your current kivy version !

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.core.window import Window
from kivy.uix.scatter import Scatter
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line, Mesh, Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.behaviors import DragBehavior
import math
import copy
import json
from PIL import Image as pilimage
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from pathlib import Path

class FlowLayouitWithCanvas(FloatLayout):

    def __init__(self, remove_poly_func, item_list, image_obj, **kwargs):
        super(FlowLayouitWithCanvas, self).__init__(**kwargs)
        self.polygon_point_list = []
        self.line_list = []
        self.point_size = (10, 10)
        self.last_scale_factor = 1.0
        self.drawing_polygon = True
        self.l = None
        self.polygons = {}
        self.item_list = item_list
        self.image_obj = image_obj
        self.remove_poly_func = remove_poly_func
        # Window.bind(mouse_pos = self.on_mouse_pos)

    def update_canvas_scale(self, scale_factor):
        scalar = scale_factor / self.last_scale_factor 
        self.point_size = (self.point_size[0]*scalar,self.point_size[1]*scalar)
        for e in self.polygon_point_list:
            e.pos = (e.pos[0]*scalar,e.pos[1]*scalar)
            e.size = (e.size[0]*scalar,e.size[1]*scalar)
        for l in self.line_list:
            l.points = [point*scalar for point in l.points]
        for polygons in self.polygons.values():
            for p_data in polygons:
                p = p_data[0]
                verticies = []
                verticies_mesh = []
                for idx in range(int((len(p.vertices)+1)/4)):
                    verticies.append([p.vertices[(idx*4)+0] * scalar,p.vertices[(idx*4)+1] * scalar])
                for verticie in verticies:
                    verticies_mesh.extend([verticie[0], verticie[1], 0 ,0])
                p.vertices = verticies_mesh
        self.last_scale_factor = scale_factor
    
    def new_image(self, new_image_name):
        if self.image_obj.source in self.polygons:
            for polygon in self.polygons[self.image_obj.source]:
                polygon_index = self.canvas.children.index(polygon[0])
                del self.canvas.children[polygon_index]
                del self.canvas.children[polygon_index-1]
                del self.canvas.children[polygon_index-2]

        if new_image_name in self.polygons:
            for polygon in self.polygons[new_image_name]: 
                self.canvas.add(Color(0, 1, 0, 0.3))
                self.canvas.add(polygon[0])
                polygon[1].color_ref = self.canvas.children[self.canvas.children.index(polygon[0])-2]

    def add_polygon(self, polygon_points, image_name, layout=None):
        points = []
        indices = []
        polygon_points_ratio = []
        if(type(polygon_points[0])==type([123.4523])):
            for point in polygon_points:
                polygon_points_ratio.append([point[0], point[1]])
                points.extend([(point[0]*self.size[0]), ((1-point[1])*self.size[1]),0,0])
                indices.append(len(indices))
        else:
            for point in polygon_points:
                polygon_points_ratio.append([(point.pos[0]+ self.point_size[0]/2)/self.size[0],(1-(point.pos[1]+self.point_size[1]/2)/self.size[1])])
                points.extend([point.pos[0]+ self.point_size[0]/2,point.pos[1]+self.point_size[1]/2,0,0])
                indices.append(len(indices))
                self.canvas.children.remove(point)
        
        polygon = Mesh(vertices=points, indices=indices, mode='triangle_fan')
        if image_name not in self.polygons:
            self.polygons[image_name] = []
        
        
        self.canvas.add(Color(0, 1, 0, 0.3))
        self.canvas.add(polygon)
        polygon_item = PolygonItem(polygon_list=self.polygons,remove_poly_func=self.remove_poly_func, points=polygon_points_ratio, image=image_name, label_name="target", polygon_ref=polygon, canvas_ref=self.canvas, size_hint=(1,None), height=50, cols=3)
        if(type(polygon_points[0])==type([123.4523])):
            polygon_item.in_dataset = True
        self.polygons[image_name].append([polygon, polygon_item])
        if(layout==None):
            layout=self.item_list
        layout.add_widget(polygon_item)

        if(image_name != self.image_obj.source):
            index = self.canvas.children.index(polygon)
            del self.canvas.children[index]
            del self.canvas.children[index-1]
            del self.canvas.children[index-2]

    def on_touch_down(self, touch):
        if touch.button == "left":
            current_pos = [touch.x- self.point_size[0]/2, touch.y- self.point_size[1]/2]
            if(self.drawing_polygon):
                dist_away = 1
                if len(self.polygon_point_list)>0:
                    first_pos = [point/self.size[idx] for idx, point in enumerate(self.polygon_point_list[0].pos)]
                    current_scaled_pos = [current_pos[0]/self.size[0],current_pos[1]/self.size[1]]
                    dist_away = math.sqrt((current_scaled_pos[0] - first_pos[0])**2 + (current_scaled_pos[1] - first_pos[1])**2)
                    if(dist_away<0.005):
                        self.add_polygon(self.polygon_point_list, self.image_obj.source, self.item_list)
                        self.drawing_polygon = False
                if(dist_away<0.005):
                    self.polygon_point_list = []
                else:
                    self.canvas.add(Color(1., 0, 0))
                    e = Ellipse(pos=((current_pos[0], current_pos[1])), size=self.point_size)
                    self.canvas.add(e)
                    self.polygon_point_list.append(e)
            elif False:
                pass
            else:
                self.drawing_polygon = True
                self.canvas.add(Color(1., 0, 0))
                e = Ellipse(pos=((current_pos[0], current_pos[1])), size=self.point_size)
                self.canvas.add(e)
                self.polygon_point_list.append(e)

    
    def clear_polygon(self):
        for point in self.polygon_point_list:
            index = self.canvas.children.index(point)
            del self.canvas.children[index]
            del self.canvas.children[index-1]
            del self.canvas.children[index-2]
        self.polygon_point_list = []
        self.canvas.add(Color(1., 0, 0))

class ZoomWindow(ScrollView):
    def __init__(self, item_list, remove_poly_func, **kwargs):
        super(ZoomWindow, self).__init__(**kwargs)
        self.i = Image(
            source='',
            size_hint=(1, 1),
            keep_ratio=True,
            allow_stretch=True
        )
        self.layout = FlowLayouitWithCanvas(size=(3000, 3000),size_hint=(None, None), remove_poly_func=remove_poly_func, item_list=item_list, image_obj=self.i)
        super(ZoomWindow, self).add_widget(self.layout)
        self.add_widget = self.layout.add_widget
        self.add_widget(self.i)

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self._keyboard.bind(on_key_up=self._on_keyboard_up)
        self.pressed_keys = set()

        self.scale_factor = 1.0
        self.scale_incremented = 0.1
        self.start_size = self.layout.size[:]

        self.point_size = (10,10)
        

    def _keyboard_closed(self):
        print('My keyboard have been closed!')
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard.unbind(on_key_up=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        self.pressed_keys.add(keycode[1])
        return True

    def _on_keyboard_up(self, keyboard, keycode):
        if(keycode[1]=="right"):
            png_list = list(Path(self.i.source).parent.glob('*.png'))
            image_idx = png_list.index(Path(self.i.source))
            if(image_idx+1==len(png_list)):
                image_idx = 0
            else:
                image_idx += 1
            self.layout.new_image(str(png_list[image_idx]))
            self.i.source = str(png_list[image_idx])
        
        elif(keycode[1]=="left"):
            png_list = list(Path(self.i.source).parent.glob('*.png'))
            image_idx = png_list.index(Path(self.i.source))
            if(image_idx==0):
                image_idx = len(png_list) - 1
            else:
                image_idx -= 1
            self.layout.new_image(str(png_list[image_idx]))
            self.i.source = str(png_list[image_idx])

        elif(keycode[1]=="escape"):
            self.layout.clear_polygon()
            
        self.pressed_keys.remove(keycode[1])
    
    def on_touch_down(self, touch):
        if "lctrl" in self.pressed_keys:
            if touch.is_mouse_scrolling:
                if touch.button == 'scrolldown':
                    self.scale_factor *= (1+self.scale_incremented)
                elif touch.button == 'scrollup':
                    self.scale_factor *= (1-self.scale_incremented)
                self.layout.size = (int(self.start_size[0]*self.scale_factor),int(self.start_size[1]*self.scale_factor))
                self.layout.update_canvas_scale(self.scale_factor)
        else:
            super(ZoomWindow, self).on_touch_down(touch)

    def on_scroll_start(self, touch, check_children=True):
        if "shift" in self.pressed_keys:
            if(touch.button=="scrollup"):
                touch.button='scrollleft'
            else:
                touch.button='scrollright'
        super(ZoomWindow, self).on_scroll_start(touch)


class ColoredLabel(Label):
    def __init__(self, bcolor=(1,1,1,1), **kwargs):
        super(ColoredLabel, self).__init__(**kwargs)
        self.bcolor = bcolor
    
    def on_size(self, *args):
        if(self.canvas != None):
            self.canvas.before.clear()
            with self.canvas.before:
                Color(self.bcolor[0],self.bcolor[1],self.bcolor[2],self.bcolor[3])
                Rectangle(pos=self.pos, size=self.size)

    def on_pos(self, *args):
        if(self.canvas != None):
            self.canvas.before.clear()
            with self.canvas.before:
                Color(self.bcolor[0],self.bcolor[1],self.bcolor[2],self.bcolor[3])
                Rectangle(pos=self.pos, size=self.size)

class PolygonItem(GridLayout):
    
    def __init__(self, polygon_list, remove_poly_func, points, image, polygon_ref, canvas_ref, label_name="test", **kwargs):
        super(PolygonItem, self).__init__(**kwargs)
        self.remove_button = Button(size_hint=(None,1),text="Remove", width=100)
        self.importance_toggle = ToggleButton(size_hint=(None,1),text="Priority", width=100)
        self.add_widget(ColoredLabel(size_hint=(1,1), text=label_name, color= (0,0,0,1), bcolor=(1,1,1,1)))
        self.add_widget(self.remove_button)
        self.add_widget(self.importance_toggle)
        self.remove_button.bind(on_press=self.button_callback)
        self.canvas_ref = canvas_ref
        self.polygon_ref = polygon_ref
        self.hover = False
        self.points = points
        self.image = image
        self.category = label_name
        self.color_ref = self.canvas_ref.children[self.canvas_ref.children.index(self.polygon_ref)-2]
        self.remove_poly_func = remove_poly_func
        self.in_dataset = False
        self.polygon_list = polygon_list
        Window.bind(mouse_pos = self.on_mouse_pos)

    def button_callback(self, instance):
        if self.polygon_ref in self.canvas_ref.children:
            self.canvas_ref.children.remove(self.polygon_ref)
        self.polygon_list[self.image].remove([self.polygon_ref, self])
        self.parent.remove_widget(self)
        if(self.in_dataset):
            self.remove_poly_func(self)

    def on_mouse_pos(self, instance, pos):
        if self.collide_point(*self.to_widget(*pos)):
            self.color_ref.r = 1
            self.hover  = True
        elif(self.hover == True):
            self.hover = False
            self.color_ref.r = 0
            

class MyApp(App):

    def __init__(self, add_polygons_func, remove_polygon_func, train_func, dataset_file, run_model_func, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        self.add_polygons_func = add_polygons_func
        self.remove_polygon_func = remove_polygon_func
        self.train_func = train_func
        self.current_polygon_batch = []
        self.dataset_file = dataset_file
        self.run_model = run_model_func

    def transfer_polygons(self, polygons, transfer_layout):
        while len(polygons)>0:
            polygon = polygons[0]
            polygon.parent.remove_widget(polygon)
            transfer_layout.add_widget(polygon)
            polygon.in_dataset = True

    def populate_polygons(self, dataset_file, scrollview, transfer_layout):
        if dataset_file.exists():
            coco_dataset_dict = {}
            with open(str(dataset_file)) as f:
                coco_dataset_dict = json.load(f)
            for annotation in coco_dataset_dict["annotations"]:
                image_name = coco_dataset_dict["images"][annotation["image_id"]]["file_name"]
                y, x = pilimage.open(image_name).size
                
                kivy_points = []
                for polygon in annotation["segmentation"]:
                    segment_iter = iter(polygon)
                    for val in segment_iter:
                        kivy_points.append([val, next(segment_iter)])
                
                scaled_points = [[point[0]/x,point[1]/y] for point in kivy_points]
                
                scrollview.layout.add_polygon(scaled_points, image_name, transfer_layout)

    def open_file(self, file):
        if(file[0].endswith('.png')):
            self.sv.layout.new_image(file[0])
            self.sv.i.source = file[0]

    def build(self):
        
        polygon_table = ScrollView(size_hint=(1,1), scroll_type=['bars'],bar_margin=10, bar_width=10)
        dataset_polygons_table = ScrollView(size_hint=(1,1), scroll_type=['bars'],bar_margin=10, bar_width=10)

        current_labels_panel = GridLayout(size_hint=(None,1), width = 300, rows=2)
        current_labels_panel.add_widget(polygon_table)
        
        dataset_labels_panel = GridLayout(size_hint=(None,1), width = 300, rows=2)
        dataset_labels_panel.add_widget(dataset_polygons_table)

        train_button = Button(size_hint=(1,None),text="Train", height=100)
        train_button.bind(on_press=lambda _: self.train_func(polygon_dataset_layout.children))

        add_polygons = Button(size_hint=(1,None),text="Add Data", height=100)
        add_polygons.bind(on_press=lambda _: self.transfer_polygons(polygon_table_layout.children, polygon_dataset_layout))
        add_polygons.bind(on_press=lambda _: self.add_polygons_func(polygon_table_layout.children))

        polygon_dataset_layout = StackLayout(orientation='lr-tb', size_hint=(1,None), spacing=5, height=4000)
        dataset_polygons_table.add_widget(polygon_dataset_layout)
    
        

        dataset_labels_panel.add_widget(train_button)
        current_labels_panel.add_widget(add_polygons)

        polygon_table_layout = StackLayout(orientation='lr-tb', size_hint=(1,None), spacing=5, height=4000)

        polygon_table.add_widget(polygon_table_layout)

        self.sv = ZoomWindow(item_list=polygon_table_layout, remove_poly_func=self.remove_polygon_func, size_hint=(1,1), scroll_type=['bars'],bar_margin=10, bar_width=10)
        self.populate_polygons(self.dataset_file, self.sv, polygon_dataset_layout)

        window = GridLayout(size=Window.size, cols=1)
        main_page = GridLayout(size=Window.size, cols=3)
        top_bar = StackLayout(orientation='lr-tb', size_hint=(1,None), spacing=5, height=50)
        file_open_btn = Button(size_hint=(None, 1), width=100, text="Open")
        top_bar.add_widget(file_open_btn)

        use_model_btn = Button(size_hint=(None, 1), width=100, text="Run")
        top_bar.add_widget(use_model_btn)
        use_model_btn.bind(on_press=lambda _: self.run_model(self.sv.i.source, self.sv.layout.add_polygon))
        
        test = GridLayout(cols=1)
        filechooser = FileChooserIconView(size_hint=(1.0,0.8), path=str(Path.home()))
        test.add_widget(filechooser)
        open_file = Button(size_hint=(1.0,0.2), text="Open")
        test.add_widget(open_file)
        test.add_widget(Label(size_hint=(0,0)))

        file_menu_window = Popup(title="File Selector", content=test, size_hint=(None, None), size=(800, 800))
        file_open_btn.bind(on_press=file_menu_window.open)
        open_file.bind(on_press=lambda _: self.open_file(filechooser.selection))

        main_page.add_widget(self.sv)
        main_page.add_widget(current_labels_panel)
        main_page.add_widget(dataset_labels_panel)
        # layout = ScalableFloatLayout(size=(3000, 3000),size_hint=(None, None))
        # layout.add_widget(l)
        window.add_widget(top_bar)
        window.add_widget(main_page)
        return window
