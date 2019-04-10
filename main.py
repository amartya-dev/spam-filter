#importing all the dependencies

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import os 
#importing the self declared modules

from classifier_import_and_predict import * #loading the best classsifier based upon prediction_scores
from classifier import * # Calculating and training three classifiers and comparing and saving the best

#=====================================================================================================
#kv file loader for the kivy app
"""The string inside the load_string method contains the widget tree. 
The present loader string uses the screen manager to toggle between 
two screens namely :
    1. Menu (containing a text input and a predict button)
    2. Result (which would show the result of the text converted and fed into the classifier)
"""

Builder.load_string("""
<MenuScreen>:
    BoxLayout:
        orientation : 'vertical'
        TextInput :
            id : text_input1
            text : 'Enter text to classify'
            font_size : 30
            height: 150
   
        Button:           
            size_hint: None, None
            pos_hint: {'center_x': .5, 'center_y' : .5}
            text: 'Predict'
            font_size : 70
            size_hint_y: None
            size_hint_x: None
            height: 150
            width : 260
            on_press: 
                root.manager.current = 'result'
                root.manager.get_screen('result').label.text = root.pred(text_input1.text)
                    
                                    
<Result>:
    label : label
    BoxLayout:
        orientation : 'vertical'
        Label : 
            id : label
            text : '1'
            font_size : 20
            size_hint_y : None
            height : 200
        Button : 
            text : 'Go Back'
            font_size : 50
            size_hint_y : None
            height : 60
            on_press : root.manager.current = 'menu'
""")

#==========================================================================================================

# Declare both screens, the python code to support and implement the above kv declarations
#The main screen
class MenuScreen(Screen):
    def pred(self,text) :
        #function called on button press
        res = predict(text) #references predict of the pre trained classifier
        return(res)

#The result screen
class Result(Screen):
    pass

#===============================================================================================

# Create the screen manager
sm = ScreenManager()
sm.add_widget(MenuScreen(name='menu'))
sm.add_widget(Result(name='result'))

class ML_ClassifierApp(App):
  
    def build(self):
        #train and pick the best classifier on app creation
        load_and_train()
        return sm
        

if __name__ == '__main__':
    ML_ClassifierApp().run()
