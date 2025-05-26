import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    from mock_gpio import GPIO

BUTTON_PIN = 17

def button_callback(channel):
    print("Button was pressed!")

def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=button_callback, bouncetime=200)

    print("Waiting for button press. Press CTRL+C to exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting program.")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()