Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptRoot
$VenvPython = Join-Path $Root "venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$AppDir = Join-Path $Root "app"
$BrandName = "Navine AI - Python"
$BrandPackage = "navine"
$BrandPort = 8766
$BrandYaml = Join-Path $Root "configs\brand.yaml"
if (Test-Path $BrandYaml) {
    foreach ($line in Get-Content $BrandYaml) {
        if ($line -match '^\s*display_name:\s*"?([^"#]+)"?\s*$') { $BrandName = $Matches[1].Trim().Trim('"') }
        if ($line -match '^\s*package:\s*"?([^"#]+)"?\s*$') { $BrandPackage = $Matches[1].Trim().Trim('"') }
        if ($line -match '^\s*port:\s*(\d+)') { $BrandPort = [int]$Matches[1] }
    }
}
$WebUrl = "http://127.0.0.1:$BrandPort"

$ColorBg = [Drawing.Color]::FromArgb(10, 14, 20)
$ColorPanel = [Drawing.Color]::FromArgb(18, 24, 32)
$ColorText = [Drawing.Color]::FromArgb(232, 237, 244)
$ColorMuted = [Drawing.Color]::FromArgb(139, 149, 165)
$ColorAccent = [Drawing.Color]::FromArgb(61, 139, 253)
$ColorAccent2 = [Drawing.Color]::FromArgb(63, 185, 80)
$ColorAccentVoice = [Drawing.Color]::FromArgb(139, 92, 246)
$ColorAccentVoice2 = [Drawing.Color]::FromArgb(56, 189, 248)
$ColorVoiceCard = [Drawing.Color]::FromArgb(14, 18, 28)
$ColorButton = [Drawing.Color]::FromArgb(22, 30, 42)
$ColorButtonBorder = [Drawing.Color]::FromArgb(42, 51, 64)
$ColorWhite = [Drawing.Color]::White

function Escape-CmdArgument {
    param([string]$Value)
    if ($null -eq $Value) { return '""' }
    return '"' + ($Value -replace '"', '""') + '"'
}

function Start-NavineCmd {
    param(
        [string]$WorkingDirectory,
        [string]$CommandLine,
        [System.Windows.Forms.FormWindowState]$WindowState = [System.Windows.Forms.FormWindowState]::Normal
    )
    $full = "cd /d $(Escape-CmdArgument $WorkingDirectory) && $CommandLine"
    if ($WindowState -eq [System.Windows.Forms.FormWindowState]::Maximized) {
        Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $full -WindowStyle Maximized | Out-Null
    } else {
        Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $full | Out-Null
    }
}

function Start-NavineCli {
    param(
        [string[]]$CliArgs,
        [System.Windows.Forms.FormWindowState]$WindowState = [System.Windows.Forms.FormWindowState]::Normal
    )
    $parts = @(
        (Escape-CmdArgument $Python),
        "-m",
        "$BrandPackage.cli"
    )
    foreach ($arg in $CliArgs) {
        $parts += Escape-CmdArgument $arg
    }
    Start-NavineCmd -WorkingDirectory $Root -CommandLine ($parts -join " ") -WindowState $WindowState
}

function Start-TrainEverything {
    $script = Join-Path $Root "scripts\train_comprehensive.py"
    if (-not (Test-Path $script)) {
        [System.Windows.Forms.MessageBox]::Show(
            "Missing scripts\train_comprehensive.py",
            $BrandName,
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        ) | Out-Null
        return
    }
    $cmd = "$(Escape-CmdArgument $Python) $(Escape-CmdArgument $script) --mode full"
    Start-NavineCmd -WorkingDirectory $Root -CommandLine $cmd -WindowState Maximized
}

function Start-QuickTrain {
    $script = Join-Path $Root "scripts\train_comprehensive.py"
    if (-not (Test-Path $script)) { return }
    $cmd = "$(Escape-CmdArgument $Python) $(Escape-CmdArgument $script) --mode quick"
    Start-NavineCmd -WorkingDirectory $Root -CommandLine $cmd -WindowState Maximized
}

function Start-CliConsole {
    $activate = if (Test-Path $VenvPython) {
        "call $(Escape-CmdArgument (Join-Path $Root "venv\Scripts\activate.bat")) && "
    } else {
        ""
    }
    $hint = "echo $BrandName CLI - type: python -m $BrandPackage.cli help && echo."
    Start-NavineCmd -WorkingDirectory $Root -CommandLine "$activate$hint"
}

function Stop-NavineServerOnPort {
    param([int]$Port = $BrandPort)
    $killed = @()
    try {
        $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $listeners) {
            $procId = $conn.OwningProcess
            if ($procId -and $procId -gt 0 -and $killed -notcontains $procId) {
                $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
                if ($proc) {
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                    $killed += $procId
                }
            }
        }
    } catch {
        $line = netstat -ano | Select-String ":$Port\s+.*LISTENING"
        foreach ($match in $line) {
            $parts = ($match -replace '\s+', ' ').ToString().Trim().Split(' ')
            if ($parts.Length -ge 5) {
                $procId = [int]$parts[-1]
                if ($procId -gt 0 -and $killed -notcontains $procId) {
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                    $killed += $procId
                }
            }
        }
    }
    return $killed
}

function Start-Server {
    param([switch]$OpenBrowser)
    $stopped = Stop-NavineServerOnPort -Port $BrandPort
    if ($stopped.Count -gt 0) {
        Start-Sleep -Milliseconds 400
        [System.Windows.Forms.MessageBox]::Show(
            "Stopped $($stopped.Count) previous $BrandName server process(es) on port $BrandPort. Starting a fresh server with the latest code.",
            $BrandName,
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
    }
    $serverCmd = "$(Escape-CmdArgument $Python) -m $BrandPackage.server"
    Start-NavineCmd -WorkingDirectory $Root -CommandLine $serverCmd
    if ($OpenBrowser) {
        Start-Sleep -Milliseconds 800
        Start-Process $WebUrl | Out-Null
    }
}

function Start-WebUI {
    Start-Server -OpenBrowser
}

function Start-ApiDocs {
    Start-Server -OpenBrowser:$false
    Start-Sleep -Milliseconds 800
    Start-Process "$WebUrl/docs" | Out-Null
}

function Start-DesktopApp {
    if (-not (Test-Path (Join-Path $AppDir "package.json"))) {
        [System.Windows.Forms.MessageBox]::Show(
            "Desktop app not found. Expected app\package.json under the project root.",
            "Navine AI - Python",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        ) | Out-Null
        return
    }
    Start-NavineCmd -WorkingDirectory $AppDir -CommandLine "npm run dev"
}

function New-SectionLabel {
    param(
        [System.Windows.Forms.Control]$Parent,
        [string]$Text,
        [int]$Y
    )
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Text
    $label.ForeColor = $ColorMuted
    $label.Location = New-Object System.Drawing.Point(16, $Y)
    $label.Size = New-Object System.Drawing.Size(388, 20)
    $label.Font = New-Object System.Drawing.Font("Segoe UI", 8.5, [System.Drawing.FontStyle]::Bold)
    $Parent.Controls.Add($label) | Out-Null
    return $label
}

function New-NavineButton {
    param(
        [System.Windows.Forms.Control]$Parent,
        [string]$Text,
        [int]$X,
        [int]$Y,
        [int]$Width,
        [int]$Height,
        [scriptblock]$OnClick,
        [System.Drawing.Color]$BackColor = $ColorButton,
        [System.Drawing.Color]$ForeColor = $ColorText
    )
    $button = New-Object System.Windows.Forms.Button
    $button.Text = $Text
    $button.Location = New-Object System.Drawing.Point($X, $Y)
    $button.Size = New-Object System.Drawing.Size($Width, $Height)
    $button.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
    $button.FlatAppearance.BorderColor = $ColorButtonBorder
    $button.FlatAppearance.BorderSize = 1
    $button.BackColor = $BackColor
    $button.ForeColor = $ForeColor
    $button.Cursor = [System.Windows.Forms.Cursors]::Hand
    $button.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $button.Add_Click($OnClick)
    $Parent.Controls.Add($button) | Out-Null
    return $button
}

function New-VoiceCard {
    param(
        [System.Windows.Forms.Control]$Parent,
        [int]$X,
        [int]$Y,
        [int]$Width,
        [int]$Height
    )
    $card = New-Object System.Windows.Forms.Panel
    $card.Location = New-Object System.Drawing.Point($X, $Y)
    $card.Size = New-Object System.Drawing.Size($Width, $Height)
    $card.BackColor = $ColorVoiceCard
    $card.Add_Paint({
        if ($null -eq $EventArgs -or $null -eq $EventArgs.Graphics) { return }
        $g = $EventArgs.Graphics
        $w = [int]$this.ClientSize.Width
        $h = [int]$this.ClientSize.Height
        $fill = New-Object System.Drawing.SolidBrush([Drawing.Color]::FromArgb(34, 28, 52))
        $g.FillRectangle($fill, 1, 1, ($w - 2), ($h - 2))
        $fill.Dispose()
        $glow = New-Object System.Drawing.Pen([Drawing.Color]::FromArgb(90, 139, 92, 246), 1)
        $g.DrawRectangle($glow, 1, 1, ($w - 3), ($h - 3))
        $glow.Dispose()
        $borderPen = New-Object System.Drawing.Pen($ColorAccentVoice, 1)
        $g.DrawRectangle($borderPen, 0, 0, ($w - 1), ($h - 1))
        $borderPen.Dispose()
        $accentPen = New-Object System.Drawing.Pen($ColorAccentVoice2, 2)
        $g.DrawLine($accentPen, 12, 0, [Math]::Min(148, ($w - 12)), 0)
        $accentPen.Dispose()
    })
    $Parent.Controls.Add($card) | Out-Null
    return $card
}

function New-HoverButton {
    param(
        [System.Windows.Forms.Control]$Parent,
        [string]$Text,
        [int]$X,
        [int]$Y,
        [int]$Width,
        [int]$Height,
        [scriptblock]$OnClick,
        [System.Drawing.Color]$BackColor = $ColorButton,
        [System.Drawing.Color]$HoverColor = $null,
        [System.Drawing.Color]$ForeColor = $ColorText
    )
    if ($null -eq $HoverColor) {
        $HoverColor = [Drawing.Color]::FromArgb(
            [Math]::Min(255, $BackColor.R + 18),
            [Math]::Min(255, $BackColor.G + 18),
            [Math]::Min(255, $BackColor.B + 18)
        )
    }
    $button = New-NavineButton -Parent $Parent -Text $Text -X $X -Y $Y -Width $Width -Height $Height -OnClick $OnClick -BackColor $BackColor -ForeColor $ForeColor
    $normal = $BackColor
    $button.Add_MouseEnter({ $button.BackColor = $HoverColor }.GetNewClosure())
    $button.Add_MouseLeave({ $button.BackColor = $normal }.GetNewClosure())
    return $button
}

function Show-InputDialog {
    param(
        [string]$Title,
        [string]$Prompt,
        [string]$DefaultValue = ""
    )
    $dialog = New-Object System.Windows.Forms.Form
    $dialog.Text = $Title
    $dialog.Size = New-Object System.Drawing.Size(460, 180)
    $dialog.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterParent
    $dialog.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dialog.MaximizeBox = $false
    $dialog.MinimizeBox = $false
    $dialog.BackColor = $ColorBg
    $dialog.ForeColor = $ColorText
    $dialog.Font = New-Object System.Drawing.Font("Segoe UI", 9.5)

    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Prompt
    $label.Location = New-Object System.Drawing.Point(16, 16)
    $label.Size = New-Object System.Drawing.Size(420, 24)
    $label.ForeColor = $ColorText
    $dialog.Controls.Add($label) | Out-Null

    $textBox = New-Object System.Windows.Forms.TextBox
    $textBox.Location = New-Object System.Drawing.Point(16, 48)
    $textBox.Size = New-Object System.Drawing.Size(412, 28)
    $textBox.Text = $DefaultValue
    $textBox.BackColor = $ColorPanel
    $textBox.ForeColor = $ColorText
    $textBox.BorderStyle = [System.Windows.Forms.BorderStyle]::FixedSingle
    $dialog.Controls.Add($textBox) | Out-Null

    $ok = New-Object System.Windows.Forms.Button
    $ok.Text = "Run"
    $ok.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $ok.Location = New-Object System.Drawing.Point(252, 96)
    $ok.Size = New-Object System.Drawing.Size(84, 32)
    $ok.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
    $ok.BackColor = $ColorAccent
    $ok.ForeColor = $ColorWhite
    $dialog.Controls.Add($ok) | Out-Null
    $dialog.AcceptButton = $ok

    $cancel = New-Object System.Windows.Forms.Button
    $cancel.Text = "Cancel"
    $cancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $cancel.Location = New-Object System.Drawing.Point(344, 96)
    $cancel.Size = New-Object System.Drawing.Size(84, 32)
    $cancel.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
    $cancel.BackColor = $ColorButton
    $cancel.ForeColor = $ColorText
    $dialog.Controls.Add($cancel) | Out-Null
    $dialog.CancelButton = $cancel

    $result = $dialog.ShowDialog()
    if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
        return $null
    }
    $value = $textBox.Text.Trim()
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $null
    }
    return $value
}

function Show-VoiceSpeakDialog {
    $dialog = New-Object System.Windows.Forms.Form
    $dialog.Text = "Speak - Navine AI - Python Voice"
    $dialog.Size = New-Object System.Drawing.Size(500, 280)
    $dialog.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterParent
    $dialog.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dialog.MaximizeBox = $false
    $dialog.MinimizeBox = $false
    $dialog.BackColor = $ColorBg
    $dialog.ForeColor = $ColorText
    $dialog.Font = New-Object System.Drawing.Font("Segoe UI", 9.5)

    $title = New-Object System.Windows.Forms.Label
    $title.Text = "Text-to-Speech"
    $title.Font = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
    $title.ForeColor = $ColorAccentVoice
    $title.Location = New-Object System.Drawing.Point(16, 12)
    $title.Size = New-Object System.Drawing.Size(460, 28)
    $dialog.Controls.Add($title) | Out-Null

    $hint = New-Object System.Windows.Forms.Label
    $hint.Text = "Type what Navine should say out loud."
    $hint.ForeColor = $ColorMuted
    $hint.Location = New-Object System.Drawing.Point(16, 40)
    $hint.Size = New-Object System.Drawing.Size(460, 20)
    $dialog.Controls.Add($hint) | Out-Null

    $textBox = New-Object System.Windows.Forms.TextBox
    $textBox.Location = New-Object System.Drawing.Point(16, 68)
    $textBox.Size = New-Object System.Drawing.Size(452, 96)
    $textBox.Multiline = $true
    $textBox.ScrollBars = [System.Windows.Forms.ScrollBars]::Vertical
    $textBox.BackColor = $ColorPanel
    $textBox.ForeColor = $ColorText
    $textBox.BorderStyle = [System.Windows.Forms.BorderStyle]::FixedSingle
    $dialog.Controls.Add($textBox) | Out-Null

    $voiceLabel = New-Object System.Windows.Forms.Label
    $voiceLabel.Text = "Voice name (optional)"
    $voiceLabel.ForeColor = $ColorMuted
    $voiceLabel.Location = New-Object System.Drawing.Point(16, 172)
    $voiceLabel.Size = New-Object System.Drawing.Size(140, 20)
    $dialog.Controls.Add($voiceLabel) | Out-Null

    $voiceBox = New-Object System.Windows.Forms.TextBox
    $voiceBox.Location = New-Object System.Drawing.Point(160, 168)
    $voiceBox.Size = New-Object System.Drawing.Size(308, 28)
    $voiceBox.BackColor = $ColorPanel
    $voiceBox.ForeColor = $ColorText
    $voiceBox.BorderStyle = [System.Windows.Forms.BorderStyle]::FixedSingle
    $dialog.Controls.Add($voiceBox) | Out-Null

    $speak = New-Object System.Windows.Forms.Button
    $speak.Text = "Speak"
    $speak.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $speak.Location = New-Object System.Drawing.Point(292, 204)
    $speak.Size = New-Object System.Drawing.Size(88, 34)
    $speak.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
    $speak.BackColor = $ColorAccentVoice
    $speak.ForeColor = $ColorWhite
    $dialog.Controls.Add($speak) | Out-Null
    $dialog.AcceptButton = $speak

    $cancel = New-Object System.Windows.Forms.Button
    $cancel.Text = "Cancel"
    $cancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $cancel.Location = New-Object System.Drawing.Point(388, 204)
    $cancel.Size = New-Object System.Drawing.Size(80, 34)
    $cancel.FlatStyle = [System.Windows.Forms.FlatStyle]::Flat
    $cancel.BackColor = $ColorButton
    $cancel.ForeColor = $ColorText
    $dialog.Controls.Add($cancel) | Out-Null
    $dialog.CancelButton = $cancel

    $result = $dialog.ShowDialog()
    if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
        return $null
    }
    $text = $textBox.Text.Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }
    $voice = $voiceBox.Text.Trim()
    if ([string]::IsNullOrWhiteSpace($voice)) {
        return @{ Text = $text }
    }
    return @{ Text = $text; Voice = $voice }
}

function Start-VoiceSpeak {
    $payload = Show-VoiceSpeakDialog
    if ($null -eq $payload) { return }
    $cliArgs = @("voice", "speak", $payload.Text)
    if ($payload.Voice) {
        $cliArgs += @("--voice", $payload.Voice)
    }
    Start-NavineCli $cliArgs
}

function Start-VoiceAdd {
    $name = Show-InputDialog -Title "Add Voice" -Prompt "Voice name:"
    if ($null -eq $name) { return }
    $sample = Show-InputDialog -Title "Add Voice" -Prompt "Path to voice sample (.wav, 6-20s):" -DefaultValue (Join-Path $Root "data\voice")
    if ($null -eq $sample) { return }
    Start-NavineCli @("voice", "add", $name, $sample)
}

function Start-VoiceTranscribe {
    $path = Show-InputDialog -Title "Transcribe Audio" -Prompt "Path to audio file (.wav):" -DefaultValue (Join-Path $Root "outputs")
    if ($null -eq $path) { return }
    Start-NavineCli @("voice", "transcribe", $path)
}

function Start-VoiceRemove {
    $name = Show-InputDialog -Title "Remove Voice" -Prompt "Voice name to remove:"
    if ($null -eq $name) { return }
    Start-NavineCli @("voice", "remove", $name)
}

function Show-VoiceDialog {
    param([System.Windows.Forms.Form]$Owner)
    Show-ScrollActionDialog -Owner $Owner -Title "Voice Studio - Navine AI - Python" -Subtitle "Speech, listening, and custom voices:" -Items @(
        @{ Label = "Voice Assistant (say Hey Navine)"; Args = @("assist", "--voice"); BackColor = $ColorAccentVoice },
        @{ Label = "Speak Text"; Action = { Start-VoiceSpeak } },
        @{ Label = "Listen (microphone)"; Args = @("voice", "listen") },
        @{ Label = "List Voices"; Args = @("voice", "list") },
        @{ Label = "Add Custom Voice"; Action = { Start-VoiceAdd } },
        @{ Label = "Remove Custom Voice"; Action = { Start-VoiceRemove } },
        @{ Label = "Transcribe Audio File"; Action = { Start-VoiceTranscribe } },
        @{ Label = "Train Voice Profiles"; Args = @("train", "voice") }
    )
}

function Show-ScrollActionDialog {
    param(
        [System.Windows.Forms.Form]$Owner,
        [string]$Title,
        [string]$Subtitle,
        [array]$Items
    )
    $dialog = New-Object System.Windows.Forms.Form
    $dialog.Text = $Title
    $dialog.Size = New-Object System.Drawing.Size(380, 520)
    $dialog.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterParent
    $dialog.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dialog.MaximizeBox = $false
    $dialog.MinimizeBox = $false
    $dialog.BackColor = $ColorBg
    $dialog.ForeColor = $ColorText
    $dialog.Font = New-Object System.Drawing.Font("Segoe UI", 9.5)

    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Subtitle
    $label.Location = New-Object System.Drawing.Point(16, 12)
    $label.Size = New-Object System.Drawing.Size(340, 24)
    $label.ForeColor = $ColorMuted
    $dialog.Controls.Add($label) | Out-Null

    $panel = New-Object System.Windows.Forms.Panel
    $panel.Location = New-Object System.Drawing.Point(0, 40)
    $panel.Size = New-Object System.Drawing.Size(364, 440)
    $panel.AutoScroll = $true
    $panel.BackColor = $ColorBg
    $dialog.Controls.Add($panel) | Out-Null

    $y = 8
    foreach ($item in $Items) {
        $localArgs = @($item.Args)
        $localWindowState = $item.WindowState
        $localAction = $item.Action
        if ($item.BackColor) {
            $backColor = $item.BackColor
            $foreColor = $ColorWhite
        } else {
            $backColor = if ($item.Accent) { $ColorAccent } else { $ColorButton }
            $foreColor = if ($item.Accent) { $ColorWhite } else { $ColorText }
        }
        if ($localAction) {
            New-NavineButton -Parent $panel -Text $item.Label -X 16 -Y $y -Width 312 -Height 32 -OnClick {
                & $localAction
                $dialog.Close()
            }.GetNewClosure() -BackColor $backColor -ForeColor $foreColor | Out-Null
        } else {
            New-NavineButton -Parent $panel -Text $item.Label -X 16 -Y $y -Width 312 -Height 32 -OnClick {
                if ($null -ne $localWindowState) {
                    Start-NavineCli -CliArgs $localArgs -WindowState $localWindowState
                } else {
                    Start-NavineCli -CliArgs $localArgs
                }
                $dialog.Close()
            }.GetNewClosure() -BackColor $backColor -ForeColor $foreColor | Out-Null
        }
        $y += 40
    }

    $dialog.ShowDialog($Owner) | Out-Null
}

function Show-ActionDialog {
    param(
        [System.Windows.Forms.Form]$Owner,
        [string]$Title,
        [string]$Subtitle,
        [array]$Items
    )
    if ($Items.Count -gt 10) {
        Show-ScrollActionDialog -Owner $Owner -Title $Title -Subtitle $Subtitle -Items $Items
        return
    }
    $dialog = New-Object System.Windows.Forms.Form
    $dialog.Text = $Title
    $dialog.Size = New-Object System.Drawing.Size(360, [Math]::Min(680, 120 + ($Items.Count * 40)))
    $dialog.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterParent
    $dialog.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $dialog.MaximizeBox = $false
    $dialog.MinimizeBox = $false
    $dialog.BackColor = $ColorBg
    $dialog.ForeColor = $ColorText
    $dialog.Font = New-Object System.Drawing.Font("Segoe UI", 9.5)

    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Subtitle
    $label.Location = New-Object System.Drawing.Point(16, 16)
    $label.Size = New-Object System.Drawing.Size(320, 24)
    $label.ForeColor = $ColorMuted
    $dialog.Controls.Add($label) | Out-Null

    $y = 48
    foreach ($item in $Items) {
        $localArgs = @($item.Args)
        $localWindowState = $item.WindowState
        $localAction = $item.Action
        if ($item.BackColor) {
            $backColor = $item.BackColor
            $foreColor = $ColorWhite
        } else {
            $backColor = if ($item.Accent) { $ColorAccent } else { $ColorButton }
            $foreColor = if ($item.Accent) { $ColorWhite } else { $ColorText }
        }
        if ($localAction) {
            New-NavineButton -Parent $dialog -Text $item.Label -X 16 -Y $y -Width 312 -Height 32 -OnClick {
                & $localAction
                $dialog.Close()
            }.GetNewClosure() -BackColor $backColor -ForeColor $foreColor | Out-Null
        } else {
            New-NavineButton -Parent $dialog -Text $item.Label -X 16 -Y $y -Width 312 -Height 32 -OnClick {
                if ($null -ne $localWindowState) {
                    Start-NavineCli -CliArgs $localArgs -WindowState $localWindowState
                } else {
                    Start-NavineCli -CliArgs $localArgs
                }
                $dialog.Close()
            }.GetNewClosure() -BackColor $backColor -ForeColor $foreColor | Out-Null
        }
        $y += 40
    }

    $dialog.ShowDialog($Owner) | Out-Null
}

function Show-ToolsDialog {
    param([System.Windows.Forms.Form]$Owner)
    Show-ScrollActionDialog -Owner $Owner -Title "Tools - Navine AI - Python" -Subtitle "System info, health, and API:" -Items @(
        @{ Label = "System Info (all models)"; Args = @("info"); Accent = $true },
        @{ Label = "Info: Text Model"; Args = @("info", "text") },
        @{ Label = "Info: Image Model"; Args = @("info", "image") },
        @{ Label = "Info: Video Model"; Args = @("info", "video") },
        @{ Label = "Doctor (health check)"; Args = @("doctor") },
        @{ Label = "Doctor + Auto Fix"; Args = @("doctor", "--fix") },
        @{ Label = "API Key: Create"; Args = @("api-key", "create") },
        @{ Label = "API Key: List"; Args = @("api-key", "list") },
        @{ Label = "Autolearn Config"; Args = @("autolearn", "config") },
        @{ Label = "Marathon Status"; Args = @("marathon", "status") },
        @{ Label = "Marathon Stop"; Args = @("marathon", "stop") }
    )
}

function Show-TrainDialog {
    param([System.Windows.Forms.Form]$Owner)
    Show-ScrollActionDialog -Owner $Owner -Title "Train - Navine AI - Python" -Subtitle "Fine-tune local models:" -Items @(
        @{ Label = "Train All Types"; Args = @("train", "all"); Accent = $true },
        @{ Label = "NSFW Text"; Args = @("train", "nsfw") },
        @{ Label = "Unrestricted Text"; Args = @("train", "unrestricted") },
        @{ Label = "Coding"; Args = @("train", "coding") },
        @{ Label = "Chat"; Args = @("train", "chat") },
        @{ Label = "Creative"; Args = @("train", "creative") },
        @{ Label = "Math"; Args = @("train", "math") },
        @{ Label = "General Text"; Args = @("train", "general") },
        @{ Label = "Multimodal Text"; Args = @("train", "multimodal-text") },
        @{ Label = "Text Model (base)"; Args = @("train", "text") },
        @{ Label = "Image Model"; Args = @("train", "image") },
        @{ Label = "Video Model"; Args = @("train", "video") },
        @{ Label = "Voice Profiles"; Args = @("train", "voice") },
        @{ Label = "Deep Train All Types"; Args = @("autolearn", "deep-train") },
        @{ Label = "Learn Chat Fine-tune"; Args = @("learn", "train") },
        @{ Label = "Image Learn + Train"; Args = @("learn", "image", "train") },
        @{ Label = "Video Learn + Train"; Args = @("learn", "video", "train") },
        @{ Label = "List Train Types"; Args = @("train", "list") }
    )
}

function Show-LearnDialog {
    param([System.Windows.Forms.Form]$Owner)
    Show-ScrollActionDialog -Owner $Owner -Title "Learn - Navine AI - Python" -Subtitle "Internet learning and autolearn:" -Items @(
        @{ Label = "NSFW Mixed Autolearn"; Args = @("learn", "nsfw", "mixed"); Accent = $true },
        @{ Label = "NSFW Mixed Loop (8 hours)"; Args = @("learn", "nsfw", "mixed-loop", "--hours", "8") },
        @{ Label = "Unrestricted Expand + Train"; Args = @("learn", "nsfw", "unrestricted-expand") },
        @{ Label = "NSFW Autolearn (full cycle)"; Args = @("learn", "nsfw", "autolearn"); Accent = $true },
        @{ Label = "NSFW Autolearn (8 hour loop)"; Args = @("learn", "nsfw", "autolearn", "--hours", "8") },
        @{ Label = "NSFW Images + Train"; Args = @("learn", "nsfw", "images") },
        @{ Label = "NSFW Videos + Train"; Args = @("learn", "nsfw", "videos") },
        @{ Label = "Unrestricted Text + Train"; Args = @("learn", "nsfw", "unrestricted") },
        @{ Label = "NSFW Local Folder"; Args = @("learn", "nsfw", "local") },
        @{ Label = "NSFW Reddit Fetch"; Args = @("learn", "nsfw", "reddit") },
        @{ Label = "NSFW 4chan Fetch"; Args = @("learn", "nsfw", "4chan") },
        @{ Label = "NSFW Waifu.im Fetch"; Args = @("learn", "nsfw", "waifu") },
        @{ Label = "NSFW Infini-Atomic Fetch"; Args = @("learn", "nsfw", "infini"); Accent = $true },
        @{ Label = "NSFW Internet Sources"; Args = @("learn", "nsfw", "sources") },
        @{ Label = "NSFW URL Fetch"; Action = { Start-NsfwCliWithUrl -Subcommand "url" -Title "NSFW URL Fetch" } },
        @{ Label = "NSFW Page Crawl"; Action = { Start-NsfwPageCrawl } },
        @{ Label = "NSFW Reddit Post"; Action = { Start-NsfwCliWithUrl -Subcommand "post" -Title "NSFW Reddit Post" } },
        @{ Label = "NSFW Status"; Args = @("learn", "nsfw", "status") },
        @{ Label = "NSFW Config"; Args = @("learn", "nsfw", "config") },
        @{ Label = "Learn Everything"; Args = @("learn", "everything") },
        @{ Label = "Learn Everything (Aggressive)"; Args = @("learn", "everything", "--aggressive") },
        @{ Label = "Coding All Sources"; Args = @("learn", "coding-all") },
        @{ Label = "All Autolearn Sources"; Args = @("learn", "all-sources") },
        @{ Label = "GitHub Code"; Args = @("learn", "github") },
        @{ Label = "Reddit Posts"; Args = @("learn", "reddit") },
        @{ Label = "Hacker News"; Args = @("learn", "hn") },
        @{ Label = "Wikipedia"; Args = @("learn", "wikipedia") },
        @{ Label = "arXiv Papers"; Args = @("learn", "arxiv") },
        @{ Label = "Public Docs Crawl"; Args = @("learn", "docs") },
        @{ Label = "Download Datasets"; Args = @("learn", "datasets") },
        @{ Label = "Ingest Learned Data"; Args = @("learn", "ingest") },
        @{ Label = "Rebuild RAG Index"; Args = @("learn", "index") },
        @{ Label = "Autolearn Cycle"; Args = @("autolearn", "run") },
        @{ Label = "Autolearn Aggressive"; Args = @("autolearn", "run", "--aggressive") },
        @{ Label = "Autolearn Loop (foreground)"; Args = @("autolearn", "start") },
        @{ Label = "Autolearn Status"; Args = @("autolearn", "status") },
        @{ Label = "Marathon (8 hours)"; Args = @("marathon", "start", "--hours", "8"); WindowState = [System.Windows.Forms.FormWindowState]::Maximized },
        @{ Label = "Marathon Status"; Args = @("marathon", "status") },
        @{ Label = "Marathon Stop"; Args = @("marathon", "stop") },
        @{ Label = "Marathon Log"; Args = @("marathon", "log") }
    )
}

function Start-PromptCli {
    param(
        [string]$Command,
        [string]$PromptTitle,
        [string]$PromptLabel,
        [string[]]$ExtraArgs = @()
    )
    $value = Show-InputDialog -Title $PromptTitle -Prompt $PromptLabel
    if ($null -eq $value) { return }
    $tokens = @($value -split '\s+')
    Start-NavineCli (@($Command) + $tokens + $ExtraArgs)
}

function Start-CustomCli {
    $value = Show-InputDialog -Title "Navine AI - Python CLI" -Prompt "Enter CLI args after navine.cli:" -DefaultValue "chat"
    if ($null -eq $value) { return }
    $tokens = $value -split '\s+'
    Start-NavineCli $tokens
}

function Start-MarathonWithHours {
    $hours = Show-InputDialog -Title "Navine AI - Python Marathon" -Prompt "How many hours?" -DefaultValue "8"
    if ($null -eq $hours) { return }
    Start-NavineCli @("marathon", "start", "--hours", $hours) -WindowState Maximized
}

function Start-NsfwAutolearnHours {
    $hours = Show-InputDialog -Title "NSFW Autolearn" -Prompt "Run NSFW autolearn for how many hours?" -DefaultValue "8"
    if ($null -eq $hours) { return }
    Start-NavineCli @("learn", "nsfw", "autolearn", "--hours", $hours) -WindowState Maximized
}

function Start-LearnUrl {
    $url = Show-InputDialog -Title "Learn URL" -Prompt "Enter URL to fetch:" -DefaultValue "https://"
    if ($null -eq $url) { return }
    Start-NavineCli @("learn", "url", $url)
}

function Start-NsfwCliWithUrl {
    param(
        [string]$Subcommand,
        [string]$Title
    )
    $url = Show-InputDialog -Title $Title -Prompt "Enter URL:" -DefaultValue "https://"
    if ($null -eq $url) { return }
    Start-NavineCli @("learn", "nsfw", $Subcommand, $url)
}

function Start-NsfwPageCrawl {
    $url = Show-InputDialog -Title "NSFW Page Crawl" -Prompt "Enter gallery or page URL:" -DefaultValue "https://"
    if ($null -eq $url) { return }
    Start-NavineCli @("learn", "nsfw", "crawl", $url)
}

[System.Windows.Forms.Application]::EnableVisualStyles()
$form = New-Object System.Windows.Forms.Form
$form.Text = $BrandName
$form.Size = New-Object System.Drawing.Size(460, 780)
$form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedSingle
$form.MaximizeBox = $false
$form.BackColor = $ColorBg
$form.ForeColor = $ColorText
$form.Font = New-Object System.Drawing.Font("Segoe UI", 9.5)

$scroll = New-Object System.Windows.Forms.Panel
$scroll.Dock = [System.Windows.Forms.DockStyle]::Fill
$scroll.AutoScroll = $true
$scroll.BackColor = $ColorBg
$form.Controls.Add($scroll) | Out-Null

$title = New-Object System.Windows.Forms.Label
$title.Text = $BrandName
$title.Font = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Bold)
$title.ForeColor = $ColorAccent
$title.Location = New-Object System.Drawing.Point(16, 14)
$title.Size = New-Object System.Drawing.Size(400, 36)
$scroll.Controls.Add($title) | Out-Null

$subtitle = New-Object System.Windows.Forms.Label
$subtitle.Text = "Your Local Multimodal AI  |  Text  Image  Video  Voice"
$subtitle.ForeColor = $ColorMuted
$subtitle.Location = New-Object System.Drawing.Point(16, 50)
$subtitle.Size = New-Object System.Drawing.Size(400, 20)
$scroll.Controls.Add($subtitle) | Out-Null

New-SectionLabel -Parent $scroll -Text "OPEN" -Y 82 | Out-Null
New-NavineButton -Parent $scroll -Text "Web UI" -X 16 -Y 104 -Width 124 -Height 38 -OnClick { Start-WebUI } -BackColor $ColorAccent -ForeColor $ColorWhite | Out-Null
New-NavineButton -Parent $scroll -Text "Desktop App" -X 148 -Y 104 -Width 124 -Height 38 -OnClick { Start-DesktopApp } | Out-Null
New-NavineButton -Parent $scroll -Text "API Server" -X 280 -Y 104 -Width 124 -Height 38 -OnClick { Start-Server } | Out-Null
New-NavineButton -Parent $scroll -Text "CLI Console" -X 16 -Y 150 -Width 192 -Height 34 -OnClick { Start-CliConsole } | Out-Null
New-NavineButton -Parent $scroll -Text "API Docs" -X 216 -Y 150 -Width 188 -Height 34 -OnClick { Start-ApiDocs } | Out-Null

New-SectionLabel -Parent $scroll -Text "CHAT & GENERATE" -Y 196 | Out-Null
New-NavineButton -Parent $scroll -Text "Chat REPL" -X 16 -Y 218 -Width 192 -Height 34 -OnClick { Start-NavineCli @("chat") } | Out-Null
New-NavineButton -Parent $scroll -Text "Capture Screen" -X 216 -Y 218 -Width 192 -Height 34 -OnClick { Start-NavineCli @("screen", "capture") } | Out-Null
New-NavineButton -Parent $scroll -Text "Text" -X 16 -Y 260 -Width 92 -Height 34 -OnClick { Start-PromptCli -Command "text" -PromptTitle "Text" -PromptLabel "Enter prompt:" } | Out-Null
New-NavineButton -Parent $scroll -Text "Image" -X 116 -Y 260 -Width 92 -Height 34 -OnClick { Start-PromptCli -Command "image" -PromptTitle "Image" -PromptLabel "Enter image prompt:" } | Out-Null
New-NavineButton -Parent $scroll -Text "Video" -X 216 -Y 260 -Width 92 -Height 34 -OnClick { Start-PromptCli -Command "video" -PromptTitle "Video" -PromptLabel "Enter video prompt:" } | Out-Null
New-NavineButton -Parent $scroll -Text "Code" -X 316 -Y 260 -Width 92 -Height 34 -OnClick { Start-PromptCli -Command "code" -PromptTitle "Code" -PromptLabel "Describe the code to generate:" } | Out-Null
New-NavineButton -Parent $scroll -Text "Custom CLI" -X 16 -Y 302 -Width 392 -Height 34 -OnClick { Start-CustomCli } | Out-Null

New-SectionLabel -Parent $scroll -Text "VOICE STUDIO" -Y 348 | Out-Null
$voiceCard = New-VoiceCard -Parent $scroll -X 16 -Y 370 -Width 392 -Height 148
$voiceTitle = New-Object System.Windows.Forms.Label
$voiceTitle.Text = "Voice assistant  |  say Hey Navine  |  TTS  STT"
$voiceTitle.Font = New-Object System.Drawing.Font("Segoe UI", 8.5, [System.Drawing.FontStyle]::Bold)
$voiceTitle.ForeColor = $ColorAccentVoice2
$voiceTitle.BackColor = [Drawing.Color]::Transparent
$voiceTitle.Location = New-Object System.Drawing.Point(14, 10)
$voiceTitle.Size = New-Object System.Drawing.Size(364, 20)
$voiceCard.Controls.Add($voiceTitle) | Out-Null
New-HoverButton -Parent $voiceCard -Text "Voice Assistant" -X 14 -Y 36 -Width 364 -Height 40 -OnClick { Start-NavineCli @("assist", "--voice") } -BackColor $ColorAccentVoice -HoverColor ([Drawing.Color]::FromArgb(167, 112, 255)) -ForeColor $ColorWhite | Out-Null
New-HoverButton -Parent $voiceCard -Text "Speak" -X 14 -Y 84 -Width 116 -Height 34 -OnClick { Start-VoiceSpeak } -BackColor ([Drawing.Color]::FromArgb(30, 36, 52)) -HoverColor ([Drawing.Color]::FromArgb(42, 50, 70)) | Out-Null
New-HoverButton -Parent $voiceCard -Text "Listen" -X 138 -Y 84 -Width 116 -Height 34 -OnClick { Start-NavineCli @("voice", "listen") } -BackColor ([Drawing.Color]::FromArgb(30, 36, 52)) -HoverColor ([Drawing.Color]::FromArgb(42, 50, 70)) | Out-Null
New-HoverButton -Parent $voiceCard -Text "All Voice" -X 262 -Y 84 -Width 116 -Height 34 -OnClick { Show-VoiceDialog -Owner $form } -BackColor $ColorAccentVoice2 -HoverColor ([Drawing.Color]::FromArgb(96, 210, 255)) -ForeColor $ColorWhite | Out-Null
New-HoverButton -Parent $voiceCard -Text "List Voices" -X 14 -Y 122 -Width 188 -Height 28 -OnClick { Start-NavineCli @("voice", "list") } -BackColor ([Drawing.Color]::FromArgb(24, 30, 44)) -HoverColor ([Drawing.Color]::FromArgb(36, 44, 60)) | Out-Null
New-HoverButton -Parent $voiceCard -Text "Train Voice" -X 210 -Y 122 -Width 168 -Height 28 -OnClick { Start-NavineCli @("train", "voice") } -BackColor ([Drawing.Color]::FromArgb(24, 30, 44)) -HoverColor ([Drawing.Color]::FromArgb(36, 44, 60)) | Out-Null

New-SectionLabel -Parent $scroll -Text "LEARN" -Y 534 | Out-Null
New-NavineButton -Parent $scroll -Text "All Learn Options" -X 16 -Y 556 -Width 192 -Height 38 -OnClick { Show-LearnDialog -Owner $form } -BackColor $ColorAccent2 -ForeColor $ColorWhite | Out-Null
New-NavineButton -Parent $scroll -Text "Learn URL" -X 216 -Y 556 -Width 192 -Height 38 -OnClick { Start-LearnUrl } | Out-Null
New-NavineButton -Parent $scroll -Text "NSFW Autolearn Now" -X 16 -Y 602 -Width 192 -Height 34 -OnClick { Start-NavineCli @("learn", "nsfw", "autolearn") -WindowState Maximized } | Out-Null
New-NavineButton -Parent $scroll -Text "NSFW Autolearn (Hours)" -X 216 -Y 602 -Width 192 -Height 34 -OnClick { Start-NsfwAutolearnHours } | Out-Null
New-NavineButton -Parent $scroll -Text "Marathon (Hours)" -X 16 -Y 644 -Width 192 -Height 34 -OnClick { Start-MarathonWithHours } | Out-Null
New-NavineButton -Parent $scroll -Text "Learn Everything" -X 216 -Y 644 -Width 192 -Height 34 -OnClick { Start-NavineCli @("learn", "everything", "--aggressive") -WindowState Maximized } | Out-Null
New-NavineButton -Parent $scroll -Text "NSFW Mixed" -X 16 -Y 686 -Width 392 -Height 34 -OnClick { Start-NavineCli @("learn", "nsfw", "mixed") -WindowState Maximized } | Out-Null

New-SectionLabel -Parent $scroll -Text "TRAIN" -Y 732 | Out-Null
New-NavineButton -Parent $scroll -Text "Train Everything" -X 16 -Y 754 -Width 192 -Height 38 -OnClick { Start-TrainEverything } -BackColor $ColorAccent -ForeColor $ColorWhite | Out-Null
New-NavineButton -Parent $scroll -Text "Quick Train" -X 216 -Y 754 -Width 192 -Height 38 -OnClick { Start-QuickTrain } | Out-Null
New-NavineButton -Parent $scroll -Text "All Train Options" -X 16 -Y 800 -Width 192 -Height 38 -OnClick { Show-TrainDialog -Owner $form } | Out-Null
New-NavineButton -Parent $scroll -Text "Train All Types" -X 216 -Y 800 -Width 192 -Height 38 -OnClick { Start-NavineCli @("train", "all") -WindowState Maximized } | Out-Null
New-NavineButton -Parent $scroll -Text "Train NSFW" -X 16 -Y 846 -Width 124 -Height 34 -OnClick { Start-NavineCli @("train", "nsfw") } | Out-Null
New-NavineButton -Parent $scroll -Text "Train Unrestricted" -X 148 -Y 846 -Width 124 -Height 34 -OnClick { Start-NavineCli @("train", "unrestricted") } | Out-Null
New-NavineButton -Parent $scroll -Text "Train Coding" -X 280 -Y 846 -Width 124 -Height 34 -OnClick { Start-NavineCli @("train", "coding") } | Out-Null
New-NavineButton -Parent $scroll -Text "Deep Train All" -X 16 -Y 888 -Width 192 -Height 34 -OnClick { Start-NavineCli @("autolearn", "deep-train") -WindowState Maximized } | Out-Null
New-NavineButton -Parent $scroll -Text "Ingest + Index" -X 216 -Y 888 -Width 192 -Height 34 -OnClick {
    $cmd = "$(Escape-CmdArgument $Python) -m $BrandPackage.cli learn ingest && $(Escape-CmdArgument $Python) -m $BrandPackage.cli learn index"
    Start-NavineCmd -WorkingDirectory $Root -CommandLine $cmd
} | Out-Null

New-SectionLabel -Parent $scroll -Text "TOOLS" -Y 934 | Out-Null
New-NavineButton -Parent $scroll -Text "Info / Status" -X 16 -Y 956 -Width 124 -Height 32 -OnClick { Start-NavineCli @("info") } | Out-Null
New-NavineButton -Parent $scroll -Text "Doctor" -X 148 -Y 956 -Width 124 -Height 32 -OnClick { Start-NavineCli @("doctor") } | Out-Null
New-NavineButton -Parent $scroll -Text "Doctor Fix" -X 280 -Y 956 -Width 124 -Height 32 -OnClick { Start-NavineCli @("doctor", "--fix") } | Out-Null
New-NavineButton -Parent $scroll -Text "All Tools" -X 16 -Y 998 -Width 192 -Height 32 -OnClick { Show-ToolsDialog -Owner $form } | Out-Null
New-NavineButton -Parent $scroll -Text "Autolearn Status" -X 216 -Y 998 -Width 192 -Height 32 -OnClick { Start-NavineCli @("autolearn", "status") } | Out-Null

New-NavineButton -Parent $scroll -Text "Exit" -X 16 -Y 1046 -Width 392 -Height 32 -OnClick { $form.Close() } | Out-Null

[void]$form.ShowDialog()
$form.Dispose()
