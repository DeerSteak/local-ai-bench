#!/bin/sh

exec /usr/bin/osascript <<'APPLESCRIPT'
set response to display dialog "Local AI Bench needs administrator permission to read power metrics." default answer "" with hidden answer buttons {"Cancel", "Continue"} default button "Continue" cancel button "Cancel" with title "Power Telemetry"
return text returned of response
APPLESCRIPT
