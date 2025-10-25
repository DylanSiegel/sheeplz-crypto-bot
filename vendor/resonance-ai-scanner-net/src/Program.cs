// Resonance.ai v12.5 Breakout Analytics (C# port)
// .NET 8 Console App
// Notes:
// - SIMPLE_MODE semantics preserved from Python: default Simple; set SIMPLE_MODE=0 to enable Pro.
// - ABS_VOL_MIN_USD respected (default 2000).
// - Coinbase public API: GET /products/{product}/candles?granularity&start&end
// - Discord webhook: plain "content" post.
//
// Build/Run:
//   dotnet new console -n ResonanceScanner
//   // replace Program.cs with this file
//   // set env as needed (Windows examples):
//   //   set SIMPLE_MODE=1
//   //   set ABS_VOL_MIN_USD=2000
//   //   set DISCORD_WEBHOOK=https://discord.com/api/webhooks/xxx/yyy
//   dotnet run
//
// Opinionated takes:
// - Kept the loop sequential to avoid Coinbase rate-limit faceplants.
// - Minimal allocations, one static HttpClient, JsonDocument parsing (fast).
// - Ctrl+C clean shutdown via CancellationToken.

using System.Diagnostics;
using System.Globalization;
using System.Text;

#nullable disable

internal sealed class Program
{


    private const string BASE_URL = "https://api.exchange.coinbase.com";
    private static readonly HttpClient http = new HttpClient
    {
        Timeout = TimeSpan.FromSeconds(15)
    };

    // ---------- Config path resolution    
    private static string ResolveConfigPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Config path is empty.");

        // 1) As given (absolute or relative to current working directory)
        var p1 = Path.GetFullPath(path, Directory.GetCurrentDirectory());
        if (File.Exists(p1)) return p1;

        // 2) Relative to the assembly base directory (bin/.../net8.0)
        var p2 = Path.GetFullPath(path, AppContext.BaseDirectory);
        if (File.Exists(p2)) return p2;

        // 3) Relative to the project root (bin/Debug/net8.0 -> project root)
        var p3 = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", path));
        if (File.Exists(p3)) return p3;

        throw new FileNotFoundException(
            $"Config not found. Tried:\n  {p1}\n  {p2}\n  {p3}");
    }

    // ---------- Loader + merger
    private static EffectiveConfig LoadConfigAndSelectMarket(string[] args)
    {

        // var configPath = GetArg(args, "--config", "-c") ?? "config.json";
        var marketName = GetArg(args, "--market", "-m") ?? "asia";

        var configArg = GetArg(args, "--config", "-c") ?? "config.json";
        var resolved = ResolveConfigPath(configArg);
        ResonanceConfig cfg;
        using (var sr = new StreamReader(resolved, Encoding.UTF8))
        using (var reader = new Newtonsoft.Json.JsonTextReader(sr))
        {
            var serializer = new Newtonsoft.Json.JsonSerializer();
            cfg = serializer.Deserialize<ResonanceConfig>(reader)
                ?? throw new InvalidOperationException("Config JSON parse failed.");
        }


        // ResonanceConfig cfg;
        // if (!string.IsNullOrWhiteSpace(configPath))
        // {
        //     if (!File.Exists(configPath)) throw new FileNotFoundException($"Config not found: {configPath}");
        //     var json = File.ReadAllText(configPath, Encoding.UTF8);
        //     cfg = JsonSerializer.Deserialize<ResonanceConfig>(json, jsonOptions)
        //           ?? throw new InvalidOperationException("Config JSON could not be parsed.");
        // }
        // else
        // {
        //     cfg = new ResonanceConfig(); // in-memory defaults
        // }

        // pick market
        if (string.IsNullOrWhiteSpace(marketName))
        {
            if (cfg.Markets.Count == 1) marketName = cfg.Markets.Keys.First();
            else throw new InvalidOperationException("No --market specified and multiple markets exist.");
        }
        if (!cfg.Markets.TryGetValue(marketName!, out var market))
            throw new InvalidOperationException($"Market '{marketName}' not found in config.");

        var marketConfig = cfg.Markets[marketName];

        // merge scanner defaults + per-market overrides
        var scanner = new ScannerConfig
        {
            Market = marketName,
            SleepSecBetweenCycles = cfg.Scanner?.SleepSecBetweenCycles ?? 10,
            CandleIntervalSec = market.Scanner?.CandleIntervalSec ?? cfg.Scanner.CandleIntervalSec,
            LookbackCandles = market.Scanner?.LookbackCandles ?? cfg.Scanner.LookbackCandles,
            AbsVolumeMinUsd = market.Scanner?.AbsVolumeMinUsd ?? cfg.Scanner.AbsVolumeMinUsd,
            SimpleMode = market.Scanner?.SimpleMode ?? cfg.Scanner.SimpleMode,
            BaseCoins = cfg.BaseCoins?.ToArray() ?? Array.Empty<string>()
        };

        return (new EffectiveConfig(scanner, marketConfig)!);
    }
    private static string GetArg(string[] args, params string[] keys)
    {
        for (int i = 0; i < args.Length; i++)
            if (keys.Contains(args[i], StringComparer.OrdinalIgnoreCase) && i + 1 < args.Length)
                return args[i + 1];
        return null;
    }
    // Effective config after merge
    private readonly record struct EffectiveConfig(ScannerConfig Scanner, MarketConfig Market, string[] BaseCoins)
    {
        public EffectiveConfig(ScannerConfig scanner, MarketConfig market)
            : this(scanner,
                   market,
                   (scanner.BaseCoins?.Length > 0 ? scanner.BaseCoins : Array.Empty<string>())
                   .Concat(market.Coins != null ? market.Coins : Array.Empty<string>())
                   .Distinct()
                   .OrderBy(s => s, StringComparer.OrdinalIgnoreCase)
                   .ToArray())
        { }
    }


    // private static readonly string[] COINS = new[]
    // {
    //     "00-USD","1INCH-USD","A8-USD","AAVE-USD","ABT-USD","ACH-USD","ACS-USD","ACX-USD","ADA-USD",
    //     "AERGO-USD","AERO-USD","AGLD-USD","AIOZ-USD","AKT-USD","ALCX-USD","ALEO-USD","ALEPH-USD",
    //     "ALGO-USD","ALICE-USD","ALT-USD","AMP-USD","ANKR-USD","APE-USD","API3-USD","APT-USD",
    //     "ARB-USD","ARKM-USD","ARPA-USD","ASM-USD","AST-USD","ATH-USD","ATOM-USD","AUCTION-USD",
    //     "AUDIO-USD","AURORA-USD","AVAX-USD","AVT-USD","AXL-USD","AXS-USD","B3-USD","BADGER-USD",
    //     "BAL-USD","BAND-USD","BAT-USD","BCH-USD","BERA-USD","BICO-USD","BIGTIME-USD","BIO-USD",
    //     "BLAST-USD","BLUR-USD","BLZ-USD","BNKR-USD","BNT-USD","BOBA-USD","BONK-USD","BTC-USD",
    //     "BTRST-USD","C98-USD","CAKE-USD","CBETH-USD","CELR-USD","CGLD-USD","CHZ-USD","CLANKER-USD",
    //     "CLV-USD","COMP-USD","COOKIE-USD","CORECHAIN-USD","COTI-USD","COW-USD","CRO-USD","CRV-USD",
    //     "CTSI-USD","CTX-USD","CVC-USD","CVX-USD","DAI-USD","DASH-USD","DEGEN-USD","DEXT-USD",
    //     "DIA-USD","DIMO-USD","DNT-USD","DOGE-USD","DOGINME-USD","DOT-USD","DRIFT-USD","EDGE-USD",
    //     "EGLD-USD","EIGEN-USD","ELA-USD","ENA-USD","ENS-USD","EOS-USD","ERA-USD","ERN-USD","ETC-USD",
    //     "ETH-USD","ETHFI-USD","FAI-USD","FARM-USD","FARTCOIN-USD","FET-USD","FIDA-USD","FIL-USD",
    //     "FIS-USD","FLOKI-USD","FLOW-USD","FLR-USD","FORT-USD","FORTH-USD","FOX-USD","FX-USD","G-USD",
    //     "GFI-USD","GHST-USD","GIGA-USD","GLM-USD","GMT-USD","GNO-USD","GODS-USD","GRT-USD","GST-USD",
    //     "GTC-USD","HBAR-USD","HFT-USD","HIGH-USD","HNT-USD","HOME-USD","HONEY-USD","HOPR-USD","ICP-USD",
    //     "IDEX-USD","ILV-USD","IMX-USD","INDEX-USD","INJ-USD","INV-USD","IO-USD","IOTX-USD","IP-USD",
    //     "JASMY-USD","JITOSOL-USD","JTO-USD","KAITO-USD","KARRAT-USD","KAVA-USD","KERNEL-USD","KEYCAT-USD",
    //     "KNC-USD","KRL-USD","KSM-USD","L3-USD","LA-USD","LCX-USD","LDO-USD","LINK-USD","LOKA-USD",
    //     "LPT-USD","LQTY-USD","LRC-USD","LRDS-USD","LSETH-USD","LTC-USD","MAGIC-USD","MANA-USD",
    //     "MANTLE-USD","MASK-USD","MATH-USD","MATIC-USD","MDT-USD","ME-USD","METIS-USD","MINA-USD",
    //     "MKR-USD","MLN-USD","MNDE-USD","MOG-USD","MOODENG-USD","MORPHO-USD","MPLX-USD","MSOL-USD",
    //     "MUSE-USD","NCT-USD","NEAR-USD","NEON-USD","NEWT-USD","NKN-USD","NMR-USD","OCEAN-USD","OGN-USD",
    //     "OMNI-USD","ONDO-USD","OP-USD","ORCA-USD","OSMO-USD","OXT-USD","PAXG-USD","PENDLE-USD",
    //     "PENGU-USD","PEPE-USD","PERP-USD","PIRATE-USD","PLU-USD","PNG-USD","PNUT-USD","POL-USD",
    //     "POLS-USD","POND-USD","POPCAT-USD","POWR-USD","PRCL-USD","PRIME-USD","PRO-USD","PROMPT-USD",
    //     "PUMP-USD","PUNDIX-USD","PYR-USD","PYTH-USD","QI-USD","QNT-USD","RAD-USD","RARE-USD","RARI-USD",
    //     "RED-USD","RENDER-USD","REQ-USD","REZ-USD","RLC-USD","RONIN-USD","ROSE-USD","RPL-USD","RSC-USD",
    //     "RSR-USD","S-USD","SAFE-USD","SAND-USD","SD-USD","SEAM-USD","SEI-USD","SHDW-USD","SHIB-USD",
    //     "SHPING-USD","SKL-USD","SKY-USD","SNX-USD","SOL-USD","SPA-USD","SPELL-USD","SPK-USD","SQD-USD",
    //     "STG-USD","STORJ-USD","STRK-USD","STX-USD","SUI-USD","SUKU-USD","SUPER-USD","SUSHI-USD",
    //     "SWELL-USD","SWFTC-USD","SXT-USD","SYRUP-USD","T-USD","TAO-USD","TIA-USD","TIME-USD","TNSR-USD",
    //     "TOSHI-USD","TRAC-USD","TRB-USD","TREE-USD","TRU-USD","TRUMP-USD","TURBO-USD","UMA-USD","UNI-USD",
    //     "USDS-USD","USDT-USD","VARA-USD","VELO-USD","VET-USD","VOXEL-USD","VTHO-USD","VVV-USD","W-USD",
    //     "WAXL-USD","WCFG-USD","WELL-USD","WIF-USD","WLD-USD","XCN-USD","XLM-USD","XRP-USD","XTZ-USD",
    //     "XYO-USD","YFI-USD","ZEC-USD","ZEN-USD","ZETA-USD","ZETACHAIN-USD","ZK-USD","ZORA-USD","ZRO-USD","ZRX-USD"
    // };

        // Verbose mode flag
private static bool VERBOSE = false;

private static bool ShouldAutoStart(string[] args)
{
    return args.Any(a => string.Equals(a, "--auto", StringComparison.OrdinalIgnoreCase) ||
                         string.Equals(a, "-y", StringComparison.OrdinalIgnoreCase));
}
private static void PrintConfigSummary(EffectiveConfig c)
{
    Console.WriteLine("\n=========== Scanner Configuration ===========");

    bool isDiscordSet = EnvironmentConfig.IsDiscordConfigured;
    
    var marketName = c.Scanner.Market.ToUpper();
    var s = c.Scanner;
    var b = c.Market.Bands;
    var totalTickers = c.Scanner.BaseCoins.Length + c.Market.Coins.Count;

    Console.WriteLine($"Market                  : {marketName}");
    Console.WriteLine($"Discord Webhook URL set? : {(isDiscordSet ? "✔" : "✘")}");
Console.WriteLine($"Discord Webhook URL: {EnvironmentConfig.GetDiscordWebhook(c.Market.DiscordWebhookEnvironmentVariable)}");


    Console.WriteLine($"Ticker pairs to scan     : {totalTickers:0}");
    Console.WriteLine($"  -Base Config           : { c.Scanner.BaseCoins.Length:0}");
    Console.WriteLine($"  -'{marketName}'  Market : { c.Market.Coins.Count:0}");


    Console.WriteLine($"\nScanner Configuration for '{marketName}' market :");
    Console.WriteLine($"  Simple Mode            : {s.SimpleMode}");
    Console.WriteLine($"  Candle Interval        : {s.CandleIntervalSec} seconds");
    Console.WriteLine($"  Lookback Candles       : {s.LookbackCandles}");
    Console.WriteLine($"  Absolute $/min volume  : {s.AbsVolumeMinUsd}");
    Console.WriteLine($"  Sleep Between Cycles   : {s.SleepSecBetweenCycles} seconds");
    Console.WriteLine($"\nBands:");
    Console.WriteLine($"  Counts                 : fast={b.Counts.Fast}, medium={b.Counts.Medium}, slow={b.Counts.Slow}");
    Console.WriteLine($"  Thresholds             : fast={b.Thresholds.Fast}, medium={b.Thresholds.Medium}, slow={b.Thresholds.Slow}");
    Console.WriteLine($"  Volume Spike Ratios    : fast={b.VolumeSpikeRatios.Fast}, medium={b.VolumeSpikeRatios.Medium}, slow={b.VolumeSpikeRatios.Slow}");
}
private static bool IsInteractive()
{
    // Good enough heuristic: if stdin is not redirected, we can prompt.
    return !Console.IsInputRedirected;
}
    private static bool ConfirmStartIfNeeded(string[] args)
    {
        if (!IsInteractive() || ShouldAutoStart(args))
            return true;

        Console.Write("\nStart scanning? [y/N]: ");
        var input = Console.ReadLine()?.Trim().ToLowerInvariant();
        return input is "y" or "yes" or "s" or "start";
    }
    private static async Task Main(string[] args)
    {

        
        // Check for --verbose arg
        VERBOSE = args.Any(a => string.Equals(a, "--verbose", StringComparison.OrdinalIgnoreCase));
        var config = LoadConfigAndSelectMarket(args);

        // Show the effective configuration
        PrintConfigSummary(config);

        // Dry-run: show config and bail (useful for CI/validation)
        if (args.Any(a => string.Equals(a, "--dry-run", StringComparison.OrdinalIgnoreCase)))
            return;

        // Human gate for testing
        if (!ConfirmStartIfNeeded(args))
        {
            Console.WriteLine("Aborting. No scanning started.");
            return;
        }

        var candleInterval = config.Scanner.CandleIntervalSec;
        var lookbackCandles = config.Scanner.LookbackCandles;
        var absVolFloor = config.Scanner.AbsVolumeMinUsd;
        var simpleMode = config.Scanner.SimpleMode;

        var bands = config.Market.Bands;
        var fastN = bands.Counts.Fast;
        var medN = bands.Counts.Medium;
        var slowN =bands.Counts.Slow;

        var thresholds = bands.Thresholds;
        var thrFast = thresholds.Fast;
        var thrMed = thresholds.Medium;
        var thrSlow = thresholds.Slow;

        var spikeRatios = bands.VolumeSpikeRatios;
        var volFast = spikeRatios.Fast;
        var volMed = spikeRatios.Medium;
        var volSlow = spikeRatios.Slow;

        Console.WriteLine("-----------------------------------------------\n");
        Console.WriteLine("\n--- Resonance.ai Breakout Scanner Activated ---");

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
            Console.WriteLine("Shutdown requested...");
        };

        var allCoinsToScan = config.Scanner.BaseCoins;
        if (allCoinsToScan is null || allCoinsToScan.Length == 0)
            throw new InvalidOperationException($"Market '{config.Market.Scanner.Market}' has no coins configured.");
            
        var marketName = config.Scanner.Market.ToUpper();
        await SendDiscordAsync($"SCANNER STARTED for {marketName}, {allCoinsToScan.Length} pairs", config.Market.DiscordWebhookEnvironmentVariable);


        //        var hdrurl = $"https://1fb297b53606.ngrok-free.app/img/BTC-USD.png";
        var hdrurl = $"https://www.tradingview.com/chart/?symbol=COINBASE:BTCUSD";
        var hdrMessage =  $"\n[See chart for BTC-USD]({hdrurl})";
                //![description](https://1fb297b53606.ngrok-free.app/img/{pair}.svg)

        await SendDiscordAsync($"{hdrMessage}", config.Market.DiscordWebhookEnvironmentVariable);

        while (!cts.IsCancellationRequested)
        {
            int successCount = 0;
            int failureCount = 0;
            var failedPairs = new List<string>();
            var noDataPairs = new List<string>();
            

            var stopwatch = new Stopwatch();
            stopwatch.Start();
            

            foreach (var pair in allCoinsToScan)
            {
                if (cts.IsCancellationRequested) break;

                try
                {
                    // Only print scanning message if verbose
                    if (VERBOSE)
                        Console.WriteLine($"Scanning {pair}...");

                    var candles = await GetCandlesAsync(pair, candleInterval, lookbackCandles, cts.Token);

                    if (candles.Count == 0)
                    {
                        Console.Write(".");
                        // Console.WriteLine($"⚠️ {pair} returned no data.");
                        noDataPairs.Add(pair);
                        int noDataPairsCount = 0;
                        noDataPairsCount++;


                        if (VERBOSE)
                            LogCoinScan(pair, null, null, null);
                        continue;
                    }

                    successCount++;
                    var startPrice = candles[0].Close;
                    var endPrice = candles[^1].Close;
                    var percentChange = (endPrice - startPrice) / startPrice * 100.0;

                    var maxHigh = candles.Max(c => c.High);
                    var minLow = candles.Min(c => c.Low);
                    var bandWidth = (maxHigh - minLow) / endPrice * 100.0;

                    var (b1, i1) = IsBreakoutBand(candles.TakeLast(fastN).ToList(), thrFast, volFast, config.Scanner.AbsVolumeMinUsd);
                    var (b2, i2) = IsBreakoutBand(candles.TakeLast(medN).ToList(), thrMed, volMed, config.Scanner.AbsVolumeMinUsd);
                    var (b3, i3) = IsBreakoutBand(candles.TakeLast(slowN).ToList(), thrSlow, volSlow, config.Scanner.AbsVolumeMinUsd);

                    var bandDetails = new List<(string name, BandStats stats)>();
                    if (b1 && i1 is not null) bandDetails.Add(("FAST", StatsFromInfo(i1)));
                    if (b2 && i2 is not null) bandDetails.Add(("MEDIUM", StatsFromInfo(i2)));
                    if (b3 && i3 is not null) bandDetails.Add(("SLOW", StatsFromInfo(i3)));

                    if (bandDetails.Count > 0)
                    {
                        Console.WriteLine($"[SELECTED] {pair} | Δ: {percentChange:F2}% | W: {bandWidth:F2}% | Hits: [{string.Join(", ", bandDetails.Select(b => b.name))}]");

                        var msg = BuildAlertMessage(
                            pair: pair,
                            price: endPrice,
                            percentChange: percentChange,
                            bandWidth: bandWidth,
                            bandDetails: bandDetails,
                            candleIntervalSec: candleInterval,
                            scannerConfig: config.Scanner);

                        // Append image URL for Discord embed
                       // var imageUrl = $"https://1fb297b53606.ngrok-free.app/img/{pair}.png";

                        var imageUrl = $"https://www.tradingview.com/chart/?symbol=COINBASE:{pair.Replace("-", "")}";



                        msg += $"\n[See chart for {pair}]({imageUrl})";
                        //![description](https://1fb297b53606.ngrok-free.app/img/{pair}.svg)
                        await SendDiscordAsync(msg, config.Market.DiscordWebhookEnvironmentVariable);
                    }
                    else
                    {
                        if (VERBOSE)
                            Console.WriteLine($"{pair} | Δ: {percentChange:F2}% | W: {bandWidth:F2}%");
                    }
                }
                catch (Exception ex)
                {
                    failureCount++;
                    failedPairs.Add(pair);
                    if (VERBOSE)
                        Console.WriteLine($"Error processing {pair}: {ex.Message}");

                    await SendDiscordAsync(ex.Message, config.Market.DiscordWebhookEnvironmentVariable);
                }
            }

            // if (failedPairs.Count > 0)
            // {
            //     var json = System.Text.Json.JsonSerializer.Serialize(failedPairs);
            //     //Console.WriteLine($"\nFailed pairs ({failedPairs.Count}): {string.Join(", ", failedPairs)}");
            //     Console.WriteLine($"\nFailed pairs:\n{json}");

            //     //using Newtonsoft.Json only, output the ones that succeeded too as a formatted json array
            //     var succeededPairs = allCoinsToScan.Except(failedPairs).ToList();
            //     var json2 = Newtonsoft.Json.JsonConvert.SerializeObject(succeededPairs, Newtonsoft.Json.Formatting.Indented);
            //     Console.WriteLine($"\nSucceeded pairs:\n{json2}");

            //     //Update allCoinsToScan to only include the succeeded pairs
            //     allCoinsToScan = succeededPairs.ToArray();

            // }
            

            // Stop and Report results via helper function
            ReportResults(stopwatch, allCoinsToScan.Length);

            Console.WriteLine($"\nCycle complete: {successCount} pairs scanned successfully.");
            if (noDataPairs.Count > 0)
            {
                Console.WriteLine($"No data on these {noDataPairs.Count} pairs: \n{string.Join(", ", noDataPairs)}");
            }
            if (failureCount > 0)
            {
                Console.WriteLine($"Failed on these {failureCount} pairs");
                Console.WriteLine($"\t {string.Join(", ", failedPairs)}");
            }
            Console.WriteLine($"Sleeping {config.Scanner.SleepSecBetweenCycles} seconds...\n");
            Console.WriteLine($"Perparing to scan {allCoinsToScan.Length} crypto pairs\n");
            try { await Task.Delay(TimeSpan.FromSeconds(config.Scanner.SleepSecBetweenCycles), cts.Token); }
            catch (TaskCanceledException)
            { /* shutdown */
                //Output the new list of coins to scan to a json file
                var outputPath = "coins_to_scan.json";
                var json = System.Text.Json.JsonSerializer.Serialize(allCoinsToScan, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                await File.WriteAllTextAsync(outputPath, json);
                Console.WriteLine($"\nShutdown complete. Updated coins to scan written to {outputPath}");

                await SendDiscordAsync("Resonance.ai Breakout Scanner. TEMPORARILY DOWN FOR MAINTENANCE.", config.Market.DiscordWebhookEnvironmentVariable);

            }
        }
    }
    static void ReportResults(Stopwatch stopwatch, int iterations)
    {
        stopwatch.Stop();
        double elapsedSeconds = stopwatch.Elapsed.TotalSeconds;

        double itemsPerSecond = iterations / elapsedSeconds;
        double itemsPerMinute = itemsPerSecond * 60;

        Console.WriteLine($"Iterations: {iterations:N0}");
        Console.WriteLine($"Elapsed Time: {elapsedSeconds:F3} seconds");
        Console.WriteLine($"Throughput: {itemsPerSecond:N2} per second");
        Console.WriteLine($"Throughput: {itemsPerMinute:N2} per minute");
    }
    private static void LogCoinScan(string symbol, double? change1h, double? bandWidth, bool? breakout)
    {
        var sb = new StringBuilder();
        sb.Append("[SKIPPED] ").Append(symbol).Append(" | ");
        if (change1h.HasValue) sb.Append($"1H Δ: {change1h.Value:F3}% | ");
        if (bandWidth.HasValue) sb.Append($"Band W: {bandWidth.Value:F3}% | ");
        if (breakout.HasValue) sb.Append("Breakout: ").Append(breakout.Value ? "✔" : "✘");
        Console.WriteLine(sb.ToString());
    }

    private static async Task<List<Candle>> GetCandlesAsync(string productId, int granularitySec, int lookbackCandles, CancellationToken ct)
    {
        try
        {
            var end = DateTimeOffset.UtcNow;
            var start = end - TimeSpan.FromSeconds(granularitySec * lookbackCandles);

            var url = $"{BASE_URL}/products/{productId}/candles" +
                      $"?granularity={granularitySec}" +
                      $"&start={Uri.EscapeDataString(start.ToString("o").Replace("+00:00", "Z"))}" +
                      $"&end={Uri.EscapeDataString(end.ToString("o").Replace("+00:00", "Z"))}";

            using var req = new HttpRequestMessage(HttpMethod.Get, url);
            req.Headers.TryAddWithoutValidation("User-Agent", "ResonanceScanner/12.5 (C#)");

            using var resp = await http.SendAsync(req, ct);
            if (!resp.IsSuccessStatusCode)
            {
                var errorContent = await resp.Content.ReadAsStringAsync(ct);
                Console.WriteLine($"❌ Error fetching candles for {productId}: HTTP {(int)resp.StatusCode}");
                Console.WriteLine($"Coinbase response: {errorContent}");
                return new List<Candle>();
            }

            var jsonString = await resp.Content.ReadAsStringAsync(ct);
            var jArray = Newtonsoft.Json.Linq.JArray.Parse(jsonString);
            if (jArray.Count == 0)
            {
                // if (jsonString.Trim() == "[]")
                //     Console.WriteLine($"⚠️ {productId} returned no data, consider removing it from the scan list.");
                // else
                // {   
                //   Console.WriteLine($"⚠️ API returned no/bad data for {productId}");
                // }    
                return new List<Candle>();
            }

            var list = new List<Candle>(jArray.Count);
            foreach (var row in jArray)
            {
                if (row is not Newtonsoft.Json.Linq.JArray arr || arr.Count < 6) continue;
                var t = arr[0].ToObject<long>();   // epoch seconds
                var low = arr[1].ToObject<double>();
                var high = arr[2].ToObject<double>();
                var open = arr[3].ToObject<double>();
                var close = arr[4].ToObject<double>();
                var volume = arr[5].ToObject<double>();
                list.Add(new Candle(t, low, high, open, close, volume));
            }

            // Coinbase returns newest-first; we want oldest-first.
            list.Sort((a, b) => a.Time.CompareTo(b.Time));
            return list;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Exception fetching candles for {productId}: {ex.Message}");
            return new List<Candle>();
        }
    }

    private static (bool hit, Dictionary<string, double> info) IsBreakoutBand(
        List<Candle> cset, double breakoutThreshold, double volumeSpikeRatio, double absVolumeMinUsd)
    {
        if (cset.Count < 3) return (false, null);

        var highs = cset.Select(c => c.High).ToArray();
        var closes = cset.Select(c => c.Close).ToArray();
        var volumes = cset.Select(c => c.Volume).ToArray();

        if (highs.Length < 2 || volumes.Length < 2) return (false, null);

        var maxHigh = highs[..^1].Max();         // highest high before last candle
        var lastClose = closes[^1];              // last close
        var avgVol = volumes[..^1].Average();
        var lastVol = volumes[^1];
        var usdPerMin = lastVol * lastClose;

        var pctOver = maxHigh > 0 ? ((lastClose / maxHigh) - 1.0) * 100.0 : 0.0;
        var volRatio = avgVol > 0 ? (lastVol / avgVol) : 0.0;

        var hit =
            lastClose > maxHigh * (1 + breakoutThreshold) &&
            lastVol > avgVol * volumeSpikeRatio &&
            usdPerMin >= absVolumeMinUsd;

        var info = new Dictionary<string, double>
        {
            ["window"] = cset.Count,
            ["last_close"] = lastClose,
            ["max_high"] = maxHigh,
            ["pct_over"] = pctOver,
            ["last_vol"] = lastVol,
            ["avg_vol"] = avgVol,
            ["vol_ratio"] = volRatio,
            ["usd_per_min"] = usdPerMin
        };

        return (hit, info);
    }

    private static BandStats StatsFromInfo(Dictionary<string, double> info)
    {
        return new BandStats(
            OverMaxPct: info.TryGetValue("pct_over", out var a) ? a : 0.0,
            Vol: info.TryGetValue("last_vol", out var b) ? b : 0.0,
            AvgVol: info.TryGetValue("avg_vol", out var c) ? c : 0.0,
            VolMult: info.TryGetValue("vol_ratio", out var d) ? d : 0.0,
            DollarsPerMin: info.TryGetValue("usd_per_min", out var e) ? e : 0.0,
            Window: (int)(info.TryGetValue("window", out var w) ? w : 0.0)
        );
    }

    private static string BuildAlertMessage(
        string pair,
        double price,
        double percentChange,
        double bandWidth,
        List<(string name, BandStats stats)> bandDetails,
        int candleIntervalSec,
        ScannerConfig scannerConfig)
    {
        if (scannerConfig != null && scannerConfig.SimpleMode)
        {
            var bandsStr = bandDetails.Count > 0
                ? string.Join(", ", bandDetails.Select(b => b.name))
                : "—";
            return string.Join('\n', new[]
            {
                "🚨 **BREAKOUT DETECTED** 🚨",
                $"**Pair**: `{pair}`",
                $"**Δ**: `{percentChange:F2}%` | **W**: `{bandWidth:F2}%`",
                $"**Bands**: {bandsStr}",
                $"**Time**: {DateTime.UtcNow:HH:mm:ss} UTC"
            });
        }
        else
        {
            var lines = new List<string>
            {
                "🚨 **BREAKOUT DETECTED** 🚨",
                $"**Pair**: `{pair}`",
                $"**Price**: `${price:F8}`",
                $"**Δ**: `{percentChange:F2}%`  |  **W**: `{bandWidth:F2}%`",
                $"**Candle**: `{candleIntervalSec}s`"
            };

            foreach (var (name, s) in bandDetails)
            {
                lines.Add(
                    $"• **{name}** (n={s.Window}): over max by `{s.OverMaxPct:F2}%`, " +
                    $"vol `{s.Vol:0}` vs avg `{s.AvgVol:0}` (x`{s.VolMult:F2}`), " +
                    $"`${s.DollarsPerMin:0}/min`"
                );
            }

            lines.Add($"**Time**: {DateTime.UtcNow:HH:mm:ss} UTC");
            return string.Join('\n', lines);
        }
    }

    private static async Task SendDiscordAsync(string message, string discordWebhookEnvironmentVariable = null)
    {

        if (EnvironmentConfig.IsDiscordConfigured == false && string.IsNullOrWhiteSpace(discordWebhookEnvironmentVariable))
        {
            Console.WriteLine("❌ Discord webhook not configured.");
            return;
        }

        string webHookUrl =  EnvironmentConfig.GetDiscordWebhook(discordWebhookEnvironmentVariable);

        try
        {
            var payload = Newtonsoft.Json.JsonConvert.SerializeObject(new { content = message });
            using var resp = await http.PostAsync(webHookUrl,
                new StringContent(payload, Encoding.UTF8, "application/json"));
            if (!resp.IsSuccessStatusCode)
            {
                Console.WriteLine($"❌ Failed to send Discord message: HTTP {(int)resp.StatusCode}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Failed to send Discord message: {ex.Message}");
        }
    }

    private readonly record struct Candle(long Time, double Low, double High, double Open, double Close, double Volume);
    private readonly record struct BandStats(double OverMaxPct, double Vol, double AvgVol, double VolMult, double DollarsPerMin, int Window);
}
