# Resonance.ai Breakout Scanner

## Credits
This is a .NET (C#) port of the original Python code by **Metteyya** (`therealmetteyya` on Discord).
- Discord Server: https://discord.gg/u3nQyEUk
- Trading Guide: [#scanner-trading-guide](https://discord.com/channels/1401761858649723002/1408580997804724316)

## Project Overview
Resonance.ai Breakout Scanner is a tool designed for crypto traders to automatically detect breakout events in digital asset prices. It scans multiple coins and markets, looking for sudden price and volume surges that may signal trading opportunities.

## How It Works
The scanner fetches recent price and volume data (candles) for each coin from Coinbase. It analyzes these candles using three different timeframes (bands: fast, medium, slow) to spot breakouts—moments when price jumps above recent highs with a spike in trading volume. When a breakout is detected, the system sends an alert to a Discord channel and prints details to the console.

## Key Features
- Multi-market and multi-coin support
- Configurable detection bands (fast, medium, slow)
- Discord webhook integration for instant alerts
- Quiet and verbose modes for console output
- Flexible configuration via JSON file

## Installation & Setup
### Prerequisites
- .NET 8 or newer
- Access to Coinbase public API
- Discord webhook URL (for alerts)

### Setup Steps
1. Clone this repository.
2. Edit `config.json` to set up your markets, coins, and detection thresholds.
3. (Optional) Set environment variables for advanced options.
4. Build and run:
   ```
   dotnet build
   dotnet run -- --market us
   ```

## Configuration
- `config.json` contains all scanner settings: which coins to scan, detection thresholds, and Discord webhook.
- You must set environment variable `DISCORD_WEBHOOK` for the notifications to reach your Discord server.
- Each market can have its own coins and detection parameters.

## Usage
- Run the scanner with command-line options:
  - `--market <name>`: (Optional) Selects which market to scan, otherwise, the first one will be used.
  - `--config <path>`: (Optional) Uses a custom config file.
  - `--verbose`: Enables detailed console output.
- Example output:
  - When a breakout is detected, you'll see details in the console and receive a Discord alert.

## Trading Concepts Used
- **Breakout:** A sudden price move above recent highs, often accompanied by high volume.
- **Volume Spike:** A sharp increase in trading volume, which can confirm the strength of a breakout.
- **Bands/Timeframes:** The scanner checks for breakouts over short, medium, and long time windows to catch different types of moves.

## Troubleshooting & FAQ
- If you see no alerts, check your config and Discord webhook.
- Make sure your .NET version is up to date.
- For API errors, verify your internet connection and Coinbase API status.

## Contributing & License
- Contributions are welcome! Please submit issues or pull requests.
- Licensed under MIT. See LICENSE for details.

## Disclaimers
- **Trading cryptocurrencies is risky.** This scanner does not guarantee profits or successful trades.
- Alerts are for informational and educational purposes only.
- Use at your own risk. Always do your own research before making investment decisions.
