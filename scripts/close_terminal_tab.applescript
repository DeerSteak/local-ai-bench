on run arguments
    set targetTTY to item 1 of arguments
    tell application "Terminal"
        repeat with terminalWindow in windows
            repeat with terminalTab in tabs of terminalWindow
                if tty of terminalTab is targetTTY then
                    close terminalTab
                    return
                end if
            end repeat
        end repeat
    end tell
end run
