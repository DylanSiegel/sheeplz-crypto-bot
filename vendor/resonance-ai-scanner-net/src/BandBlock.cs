public sealed class BandBlock
{
    public Counts Counts { get; init; } = new();
    public Thresholds Thresholds { get; init; } = new();
    public VolumeSpikeRatios VolumeSpikeRatios { get; init; } = new();
}

public sealed class Counts { public int Fast { get; init; } = 12; public int Medium { get; init; } = 18; public int Slow { get; init; } = 24; }
public sealed class Thresholds { public double Fast { get; init; } = 0.014; public double Medium { get; init; } = 0.018; public double Slow { get; init; } = 0.026; }
public sealed class VolumeSpikeRatios { public double Fast { get; init; } = 1.5; public double Medium { get; init; } = 1.8; public double Slow { get; init; } = 2.6; }
