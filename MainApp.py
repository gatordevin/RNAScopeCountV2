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
from kivy.core.window import Window
from kivy.uix.scatter import Scatter
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.behaviors import DragBehavior

class FlowLayouitWithCanvas(FloatLayout):

    def __init__(self, **kwargs):
        super(FlowLayouitWithCanvas, self).__init__(**kwargs)
        self.polygon_point_list = []
        self.line_list = []
        self.point_size = (15, 15)
        self.last_scale_factor = 1.0
        self.drawing_polygon = False
        self.l = None
        Window.bind(mouse_pos = self.on_mouse_pos)

    def update_elipse_pos_scale(self, scale_factor):
        scalar = scale_factor / self.last_scale_factor 
        self.point_size = (self.point_size[0]*scalar,self.point_size[1]*scalar)
        for e in self.polygon_point_list:
            e.pos = (e.pos[0]*scalar,e.pos[1]*scalar)
            e.size = (e.size[0]*scalar,e.size[1]*scalar)
        for l in self.line_list:
            l.points = [point*scalar for point in l.points]
        self.last_scale_factor = scale_factor
    
    def on_touch_down(self, touch):
        if touch.button == "left":
            self.drawing_polygon = True
            Color(1., 0, 0)
            self.canvas.add(Color(1., 0, 0))
            e = Ellipse(pos=((touch.x - self.point_size[0]/2, touch.y - self.point_size[1]/2)), size=self.point_size)
            print(touch.x,touch.y)
            self.canvas.add(e)
            l = Line(points=(e.pos[0],e.pos[1]))
            self.line_list.append(l)
            self.canvas.add(l)
            self.polygon_point_list.append(e)
    
    def on_mouse_pos(self, instance, pos):
        # print(instance.to_widget(*pos))
        # pos.apply_transform_2d(self.to_local)
        if self.line_list != []:
            points = self.line_list[-1].points
            self.line_list[-1].points = (points[0], points[1], pos[0], pos[1])

class ZoomWindow(ScrollView):
    def __init__(self, **kwargs):
        super(ZoomWindow, self).__init__(**kwargs)
        self.layout = FlowLayouitWithCanvas(size=(3000, 3000),size_hint=(None, None))
        super(ZoomWindow, self).add_widget(self.layout)
        self.add_widget = self.layout.add_widget

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
        self.pressed_keys.remove(keycode[1])
    
    def on_touch_down(self, touch):
        if "lctrl" in self.pressed_keys:
            if touch.is_mouse_scrolling:
                if touch.button == 'scrolldown':
                    self.scale_factor *= (1+self.scale_incremented)
                elif touch.button == 'scrollup':
                    self.scale_factor *= (1-self.scale_incremented)
                self.layout.size = (int(self.start_size[0]*self.scale_factor),int(self.start_size[1]*self.scale_factor))
                self.layout.update_elipse_pos_scale(self.scale_factor)
        else:
            super(ZoomWindow, self).on_touch_down(touch)

    def on_scroll_start(self, touch, check_children=True):
        if "shift" in self.pressed_keys:
            if(touch.button=="scrollup"):
                touch.button='scrollleft'
            else:
                touch.button='scrollright'
        super(ZoomWindow, self).on_scroll_start(touch)


class MyApp(App):

    def build(self):
        sv = ZoomWindow(size=Window.size, scroll_type=['bars'],bar_margin=10, bar_width=10)
        l = Image(
            source='mCherry-0003.png',
            size_hint=(1, 1),
            keep_ratio=True,
            allow_stretch=True
        )
        sv.add_widget(l)
        # layout = ScalableFloatLayout(size=(3000, 3000),size_hint=(None, None))
        # layout.add_widget(l)
        
        return sv