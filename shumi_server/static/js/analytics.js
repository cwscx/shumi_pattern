// static/js/analytics.js
export function getRollingMilkTotal(patterns, targetMl) {
    const now = new Date();
    const twentyFourHoursAgo = new Date(now.getTime() - (24 * 60 * 60 * 1000));
    let total = 0;

    patterns.forEach(day => {
        // day.date is "2026/01/11"
        day.actions.forEach(action => {
            if (action.action === '喝奶' && action.volume) {
                // Combine date and time to get absolute timestamp
                const actionTime = new Date(`${day.date.replaceAll('/', '-')}T${action.time_start}`);
                
                if (actionTime >= twentyFourHoursAgo && actionTime <= now) {
                    const volume = parseInt(action.volume.replace('ml', ''));
                    total += volume;
                }
            }
        });
    });

    updateHydrationUI(total, targetMl);
    return total;
}

function updateHydrationUI(total, target) {
    const display = document.getElementById('rollingMilkDisplay');
    const liquid = document.getElementById('liquidLevel');
    const advice = document.getElementById('hydrationAdvice');

    const percent = Math.min((total / target) * 100, 100);
    
    display.innerHTML = `${total} <span class="unit">ml</span>`;
    liquid.style.height = `${percent}%`;

    if (percent < 30) {
        advice.innerText = "⚠️ 奶量偏低，飞机上空气干燥，请注意补水。";
        advice.style.color = "#ff4b2b";
    } else if (percent < 70) {
        advice.innerText = "🥤 补水进度正常，继续保持。";
        advice.style.color = "#ffa502";
    } else {
        advice.innerText = "✅ 摄入充足！舒米现在状态很棒。";
        advice.style.color = "#2ed573";
    }
}