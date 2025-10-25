#nullable disable
public static class EnvironmentConfig
{
    static string DiscordWebhook { get; } = Environment.GetEnvironmentVariable("DISCORD_WEBHOOK_URL");

    static EnvironmentConfig()
    {
        DiscordWebhook = Environment.GetEnvironmentVariable("DISCORD_WEBHOOK_URL")
            ?? throw new InvalidOperationException("Missing DISCORD_WEBHOOK env var.");

    }
    public static bool IsDiscordConfigured => !string.IsNullOrWhiteSpace(DiscordWebhook);
    public static string GetDiscordWebhook(string envVarName)
    {
        envVarName = envVarName?.Trim() ?? throw new ArgumentNullException(envVarName);
        var x = Environment.GetEnvironmentVariable(envVarName);
        return x
            ?? DiscordWebhook;
    }
}
