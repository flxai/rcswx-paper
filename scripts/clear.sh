#!/bin/bash

# a script to detach, and kill all the screens
# useful when you have a lot of screens running
# and you want to clear them all
for session in $(screen -ls | grep -o '[0-9]*\.' | grep -o '[0-9]*'); do
    # stop any code running
    screen -S "${session}" -X stuff "^C"
    # kill the screen
    screen -S "${session}" -X quit
done
