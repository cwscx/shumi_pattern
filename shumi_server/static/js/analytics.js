// static/js/analytics.js
let milkPieChart = null; // Global to this module to allow for updates

export function getRollingMilkTotal(patterns, targetMl) {
    const now = new Date();
    const twentyFourHoursAgo = now.getTime() - (24 * 60 * 60 * 1000);
    
    let total = 0;
    let formulaTotal = 0;
    let breastMilkTotal = 0;

    patterns.forEach(day => {
        const cleanDate = day.date.replaceAll('/', '-');
        day.actions.forEach(action => {
            if (action.action === '喝奶' && action.volume) {
                const actionTime = new Date(`${cleanDate} ${action.time_start}`);
                
                if (!isNaN(actionTime) && actionTime.getTime() >= twentyFourHoursAgo && actionTime.getTime() <= now.getTime()) {
                    const volume = parseInt(action.volume.replace(/[^0-9]/g, ''));
                    total += volume;

                    // Categorize based on subType
                    if (action.type === '配方奶') {
                        formulaTotal += volume;
                    } else {
                        // Includes 瓶喂母乳 and 亲喂母乳
                        breastMilkTotal += volume;
                    }
                }
            }
        });
    });

    updateHydrationUI(total, targetMl);
    updateMilkPieChart(formulaTotal, breastMilkTotal);
    return total;
}


function updateHydrationUI(total, target) {
    const display = document.getElementById('rollingMilkDisplay');
    const liquid = document.getElementById('liquidLevel');
    const advice = document.getElementById('hydrationAdvice');
    const targetDisplay = document.getElementById('milkTargetDisplay');
    if (targetDisplay) {
        targetDisplay.innerText = target;
    }
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

function updateMilkPieChart(formula, breast) {
    const ctx = document.getElementById('milkTypeChart').getContext('2d');
    
    if (milkPieChart) milkPieChart.destroy();

    milkPieChart = new Chart(ctx, {
        type: 'doughnut', // Doughnut looks cleaner than a full pie
        data: {
            labels: ['配方奶', '母乳'],
            datasets: [{
                data: [formula, breast],
                backgroundColor: ['#ff9f43', '#54a0ff'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } }
            },
            cutout: '70%' // Makes it a thin ring
        }
    });
}