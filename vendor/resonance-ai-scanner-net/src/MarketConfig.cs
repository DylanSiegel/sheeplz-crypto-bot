public sealed class MarketConfig
{
    public ScannerConfig? Scanner { get; init; } // optional per-market overrides
    public BandBlock Bands { get; init; } = new();

    //The list of coins to scan in this market. These coins are specicifc to each market
    //and will be appended to the base list of coins in the config.json file
    //If empty, only the base list of coins will be scanned.
    public List<string> Coins { get; init; } = new();
    public string DiscordWebhookEnvironmentVariable { get; init; } = "DISCORD_WEBHOOK_URL";
}