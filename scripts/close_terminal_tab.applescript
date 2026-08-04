on run arguments
    set targetTTY to item 1 of arguments
    delay 0.2
    tell application "Terminal"
        repeat with terminalWindow in windows
            repeat with terminalTab in tabs of terminalWindow
                if tty of terminalTab is targetTTY then
                    if (count of tabs of terminalWindow) is 1 then
                        close terminalWindow
                    else
                        set selected of terminalTab to true
                        activate
                        tell application "System Events" to keystroke "w" using command down
                    end if
                    return
                end if
            end repeat
        end repeat
    end tell
    error "The Terminal session could not be found." number 1
end run
