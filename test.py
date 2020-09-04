from kivy.config import Config
Config.set('graphics', 'resizable', '1') 
Config.set('graphics', 'width', '1920') 
Config.set('graphics', 'height', '1080')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

import kivy
kivy.require('2.0.0')
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Ellipse, Line

class ImageAnnotate(Image):
    scale_factor = 1.0
    
    def __init__(self, **kwargs):
        super(ImageAnnotate, self).__init__(**kwargs)
        self.elipse_list = []

        self.bind(size=self.update_rect)

    def update_rect(self, *args):
        for elipse in self.elipse_list:
            scale = args[1][0] / self.texture_size[0]
            print(scale)
            elipse.pos = [elipse.pos[0]*scale,elipse.pos[1]*scale]
    
    def on_touch_down(self, touch):
        e = Ellipse(pos=(touch.x, touch.y), size=(10,10))
        self.canvas.add(e)
        self.elipse_list.append(e)
    
    def scale(self, increment):
        increment = (1+increment)
        print(self.scale_factor)
        self.size = (int(self.size[0]*increment),int(self.size[1]*increment))
        for elipse in self.elipse_list:
            elipse.pos = [elipse.pos[0]*increment,elipse.pos[1]*increment]
        
class ScrollViewWithZoom(ScrollView):
    image = ObjectProperty(None)
    scale_factor = 1.0
    def __init__(self, **kwargs):
        super(ScrollViewWithZoom, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self._keyboard.bind(on_key_up=self._on_keyboard_up)
        self.pressed_keys = set()
        self.scale_incremented = 0.1
        # self.start_size = self.image.size[:]

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
        image = ObjectProperty()

    def on_touch_down(self, touch):
        if "lctrl" in self.pressed_keys:
            if touch.is_mouse_scrolling:
                if touch.button == 'scrolldown':
                    # self.scale_factor += self.scale_incremented
                    self.image.scale(self.scale_incremented)
                elif touch.button == 'scrollup':
                    # self.scale_factor -= self.scale_incremented
                    self.image.scale(-self.scale_incremented)
                # self.image.size = (int(self.start_size[0]*self.scale_factor),int(self.start_size[1]*self.scale_factor))
                # print(self.scale_factor)
        else:
            super(ScrollViewWithZoom, self).on_touch_down(touch)
    
    def on_scroll_start(self, touch, check_children=True):
        if "shift" in self.pressed_keys:
            if(touch.button=="scrollup"):
                touch.button='scrollleft'
            else:
                touch.button='scrollright'
        super(ScrollViewWithZoom, self).on_scroll_start(touch)

    
class ControllerApp(App):

    def build(self):
        return ScrollViewWithZoom()


if __name__ == '__main__':
    ControllerApp().run()