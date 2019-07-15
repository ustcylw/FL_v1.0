# FL_v1.0
detect facial landmark with hourglass



# test nmse
FL_315_58160_model.ckpt
error_per_landmark:  2831.8599271209087
Result: 
mean error: 4.75943 
failures(err>0.1): 9.08%(54/595)
 

# speed test：
linux-i7(cpu)
speed: 0.07510571119164218 44.6878981590271

linux-i7-gpu(2080)
speed: 0.01053304592100512 6.267162322998047

