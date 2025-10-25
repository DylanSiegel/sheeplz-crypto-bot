
/// <summary>
/// Resonance configuration root object in the config.json file.
/// </summary>
public sealed class ResonanceConfig
{
    /// <summary>
    /// Scanner configuration section. 
    /// </summary>
    public ScannerConfig Scanner { get; init; } = new();

    /// <summary>
    /// Market configurations section, keyed by market name (e.g. "us", "eu", "asia").  Each market
    /// </summary>
    public Dictionary<string, MarketConfig> Markets { get; init; } = new();

    /// <summary>
    /// List of base coins to scan.  These will be combined with each coin in the selected market's coin list. 
    ///  If empty, only the market's coin list will be scanned.
    /// </summary>
    public List<string> BaseCoins { get; init; } = new();


}