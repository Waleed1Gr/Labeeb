class MotorController:
    def __init__(self, motor_driver):
        self.motor_driver = motor_driver

    def set_speed(self, speed):
        """
        Set the speed of the motor.
        :param speed: Speed value (e.g., -100 to 100)
        """
        speed = max(min(speed, 100), -100)
        self.motor_driver.set_pwm(abs(speed))
        if speed > 0:
            self.motor_driver.forward()
        elif speed < 0:
            self.motor_driver.backward()
        else:
            self.motor_driver.stop()

    def stop(self):
        """Stop the motor."""
        self.motor_driver.stop()


# Example motor driver interface (to be implemented for your hardware)
class ExampleMotorDriver:
    def set_pwm(self, value):
        print(f"Setting PWM to {value}")

    def forward(self):
        print("Motor moving forward")

    def backward(self):
        print("Motor moving backward")

    def stop(self):
        print("Motor stopped")


# Example usage
if __name__ == "__main__":
    driver = ExampleMotorDriver()
    motor = MotorController(driver)
    motor.set_speed(50)
    motor.set_speed(-30)
    motor.stop()