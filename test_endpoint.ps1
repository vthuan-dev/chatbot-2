# PowerShell script to test RunPod endpoint
$API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
$ENDPOINT_ID = "r9es7ffu9rq9iu"

$headers = @{
    "Content-Type" = "application/json"
    "Authorization" = "Bearer $API_KEY"
}

$body = @{
    input = @{
        prompt = "Xin chào, cho tôi biết về Đại học Cần Thơ?"
        max_tokens = 256
        temperature = 0.7
    }
} | ConvertTo-Json

$url = "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync"

Write-Host "Testing CTU Chatbot endpoint..." -ForegroundColor Green
Write-Host "Sending request to: $url" -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod -Uri $url -Method Post -Headers $headers -Body $body
    
    if ($response.status -eq "COMPLETED") {
        Write-Host "`nResponse:" -ForegroundColor Green
        Write-Host $response.output.response
        Write-Host "`nTime: $($response.output.response_time)s" -ForegroundColor Cyan
        Write-Host "Tokens: $($response.output.tokens_generated)" -ForegroundColor Cyan
    } else {
        Write-Host "Status: $($response.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
