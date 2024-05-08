#import ctypes 
import ctypes 
  

x =  '0x0000020F0B8851C0'
  
# display memory address 
print("Memory address - ", x) 
  
# get the value through memory address 
a = ctypes.cast(x, ctypes.py_object).value 
  
# display 
print("Value - ", a) 