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

let formulaTrendChart = null;


// static/js/analytics.js

// static/js/analytics.js

export function initFormulaTrend(patterns) {
    const ctx = document.getElementById('formulaTrendChart').getContext('2d');
    
    const labels = [];
    const formulaData = [];
    const breastMilkData = [];

    // Sort patterns by date
    const sortedPatterns = [...patterns].sort((a, b) => new Date(a.date) - new Date(b.date));

    sortedPatterns.forEach(day => {
        let dayFormula = 0;
        let dayBreast = 0;

        day.actions.forEach(action => {
            if (action.action === '喝奶' && action.volume) {
                const vol = parseInt(action.volume.replace(/[^0-9]/g, ''));
                if (!isNaN(vol)) {
                    if (action.type === '配方奶') {
                        dayFormula += vol;
                    } else {
                        dayBreast += vol;
                    }
                }
            }
        });

        // Only add dates where Shumi actually drank something
        if (dayFormula + dayBreast > 0) {
            labels.push(day.date);
            formulaData.push(dayFormula);
            breastMilkData.push(dayBreast);
        }
    });

    window.formulaTrendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '配方奶 (ml)',
                    data: formulaData,
                    borderColor: '#ff9f43',
                    backgroundColor: 'rgba(255, 159, 67, 0.6)',
                    fill: true,
                    stacked: true, // This tells Chart.js to stack this on top of the next one
                    tension: 0.3,
                    pointRadius: 3
                },
                {
                    label: '母乳 (ml)',
                    data: breastMilkData,
                    borderColor: '#54a0ff',
                    backgroundColor: 'rgba(84, 160, 255, 0.6)',
                    fill: true,
                    stacked: true,
                    tension: 0.3,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    stacked: true, // This is the "Magic" that makes the total height = formula + breast
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '每日总奶量 (ml)'
                    },
                    ticks: {
                        callback: v => v + "ml"
                    },
                    grid: {
                        color: (context) => (context.tick.value === 800 ? '#ff4b2b' : '#ececec'),
                        lineWidth: (context) => (context.tick.value === 800 ? 2 : 1),
                    }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        footer: (tooltipItems) => {
                            let sum = 0;
                            tooltipItems.forEach(i => sum += i.parsed.y);
                            return `总计: ${sum}ml`;
                        }
                    }
                }
            }
        }
    });
}


export function getRollingSleepTotal(patterns) {
    const now = new Date();
    const twentyFourHoursAgo = now.getTime() - (24 * 60 * 60 * 1000);
    let totalMinutes = 0;

    patterns.forEach(day => {
        const cleanDate = day.date.replaceAll('/', '-');
        day.actions.forEach(action => {
            if (action.action === '睡眠' && action.time_start) {
                const start = new Date(`${cleanDate}T${action.time_start}`).getTime();
                // If sleep is ongoing, use 'now' as the end time
                const end = action.time_end 
                    ? new Date(`${cleanDate}T${action.time_end}`).getTime() 
                    : now.getTime();

                // Check if this sleep session overlaps with our 24h window
                const overlapStart = Math.max(start, twentyFourHoursAgo);
                const overlapEnd = Math.min(end, now.getTime());

                if (overlapEnd > overlapStart) {
                    totalMinutes += (overlapEnd - overlapStart) / (1000 * 60);
                }
            }
        });
    });

    const totalHours = (totalMinutes / 60).toFixed(1);
    updateSleepUI(totalHours);
    return totalHours;
}

function updateSleepUI(hours) {
    const display = document.getElementById('rollingSleepDisplay');
    const advice = document.getElementById('sleepAdvice');
    const moonIcon = document.getElementById('sleepMoon');

    if (!display) return;

    display.innerHTML = `${hours} <span class="unit">h</span>`;

    // Sleep Health Logic for a 5-month old (Target 12-14h total per 24h)
    if (hours < 9) {
        advice.innerText = "⚠️ 睡眠严重不足。可能会非常烦躁，建议优先安排补觉。";
        advice.style.color = "#ff4b2b";
        moonIcon.style.color = "#ff4b2b";
    } else if (hours < 12) {
        advice.innerText = "😴 累计睡眠稍低。时差调整期间正常，尽量维持小睡。";
        advice.style.color = "#ffa502";
        moonIcon.style.color = "#ffa502";
    } else {
        advice.innerText = "✅ 睡眠充足。舒米正在很好地适应新节奏！";
        advice.style.color = "#2ed573";
        moonIcon.style.color = "#2ed573";
    }
}

// static/js/analytics.js
export function initBioClockChart(patterns) {
    const ctx = document.getElementById('bioClockChart').getContext('2d');
    const labels = [];
    const bedtimeData = [];

    // Sort history by date
    const sortedPatterns = [...patterns].sort((a, b) => new Date(a.date) - new Date(b.date));

    sortedPatterns.forEach(day => {
        // Find the "Night Anchor": The first sleep block of the night.
        // We look for the first sleep session starting after 5:00 PM (17:00) 
        // OR very early morning (before 4:00 AM) if she stayed up late.
        const nightSleepCandidates = day.actions.filter(a => {
            if (a.action !== '睡眠') return false;
            const [hrs] = a.time_start.split(':').map(Number);
            return (hrs >= 17 || hrs <= 4);
        }).sort((a, b) => {
            // Logic: 17:00 comes first, 04:00 comes last
            const getVal = h => h <= 4 ? h + 24 : h;
            return getVal(parseInt(a.time_start)) - getVal(parseInt(b.time_start));
        });

        if (nightSleepCandidates.length > 0) {
            const firstSleep = nightSleepCandidates[0];
            const [hrs, mins] = firstSleep.time_start.split(':').map(Number);
            
            // Decimal conversion for the Y-axis
            let decimalTime = hrs + (mins / 60);
            
            // To keep the line continuous on the 0-24 scale:
            // If she falls asleep at 1:00 AM, it's 1.0. 
            // If she falls asleep at 8:00 PM, it's 20.0.
            labels.push(day.date);
            bedtimeData.push(decimalTime);
        }
    });

    window.bioClockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: '首个长睡眠开始时间 (PST)',
                data: bedtimeData,
                borderColor: '#6c5ce7',
                backgroundColor: 'rgba(108, 92, 231, 0.1)',
                borderWidth: 3,
                pointRadius: 4,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0, 
                    max: 24,
                    ticks: {
                        stepSize: 2,
                        callback: v => v + ":00"
                    },
                    title: { display: true, text: '24小时制 (California PST)' }
                },
                x: { grid: { display: false } }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            let v = ctx.parsed.y;
                            let h = Math.floor(v);
                            let m = Math.round((v - h) * 60);
                            return `加州时间: ${h}:${m.toString().padStart(2, '0')} PST`;
                        }
                    }
                }
            }
        }
    });
}