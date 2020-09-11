from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.button import Button

class MyWidget(GridLayout):


    def selected(self, filename):
        try:
            self.ids.image.source = filename[0]


        except:
            pass



class FileChooserWindow(App):
    def test(self, filename):
        print(filename)

    def build(self):
        test = MyWidget(cols=2)
        filechooser = FileChooserIconView(size_hint=(0.5,0.5))
        train_button = Button(size_hint=(1,None),text="Train", height=100)
        train_button.bind(on_press=lambda _: self.test(filechooser.selection))
        test.add_widget(filechooser)
        test.add_widget(train_button)
        return test




if __name__ == "__main__":
    window = FileChooserWindow()
    window.run()