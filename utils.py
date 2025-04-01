
def make_motortime_shorter(sensor_time, motor_time):
    if motor_time > sensor_time + 1000:
        motor_time = motor_time - 1000

    # der sensor fällt zwar vor dem motor aus, aber weniger als 1000 stunden vor dem motor
    if motor_time > sensor_time and sensor_time + 1000 > motor_time:
        motor_time = sensor_time

    return motor_time


if __name__ == "__main__":
    # Fall Motor-Zeit wird verkürzt 1500
    print(make_motortime_shorter(sensor_time=1000, motor_time= 2500))


    # Fall motor kommt vor sensor
    print(make_motortime_shorter(sensor_time = 2000, motor_time = 1000))

    # Sensor direkt Motor 
    print(make_motortime_shorter(sensor_time = 2000, motor_time = 2500))