public sealed class ScannerConfig
{
    public string Market { get; set; } = "us";
    public int CandleIntervalSec { get; set; } = 60;
    public int LookbackCandles { get; set; } = 10;
    public double AbsVolumeMinUsd { get; set; } = 2000;
    public bool SimpleMode { get; set; } = true;
    public int SleepSecBetweenCycles { get; set; } = 10;
    public string[] BaseCoins { get; set; } = Array.Empty<string>();
    
}