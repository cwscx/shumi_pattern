// static/js/chart_logic.js

// static/js/chart_logic.js

let chartInstance = null;
let originalData = null;

export function initChart() {
    const dataElement = document.getElementById('chart-data');
    if (!dataElement) return;
    const rawData = JSON.parse(dataElement.textContent);
    renderChart(rawData);
}

function renderChart(rawData) {
    const ctx = document.getElementById('routineChart').getContext('2d');
    if (chartInstance) chartInstance.destroy(); // Clean up old chart

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            datasets: [
                {
                    label: '😴 睡眠',
                    data: rawData.sleep,
                    backgroundColor: 'rgba(156, 39, 176, 0.7)',
                    borderColor: '#9c27b0',
                    borderWidth: 1,
                    borderRadius: 4,
                    grouped: false // Keeps all blocks in the same column
                },
                {
                    label: '🍼 喝奶',
                    data: rawData.milk,
                    backgroundColor: 'rgba(33, 150, 243, 0.7)',
                    borderColor: '#2196f3',
                    borderWidth: 1,
                    borderRadius: 4,
                    grouped: false
                },
                {
                    label: '🧷 尿布',
                    data: rawData.diaper,
                    backgroundColor: 'rgba(255, 152, 0, 0.7)',
                    borderColor: '#ff9800',
                    borderWidth: 1,
                    borderRadius: 4,
                    grouped: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false, // Required to respect the container's height
            scales: {
                x: {
                    title: { display: true, text: '观察日期' },
                    grid: { display: false } // Cleans up the look
                },
                y: {
                    min: 0,
                    max: 24,
                    reverse: true, 
                    title: { display: true, text: '时间 (点击柱状块查看详情)' },
                    ticks: {
                        stepSize: 1, // Show a label for every single hour
                        autoSkip: false, // Force all hour labels to show
                        callback: value => {
                            // Formats 0-24 into 00:00 - 24:00
                            return value.toString().padStart(2, '0') + ":00";
                        },
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: '#ececec' // Light grid lines for every hour
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
}


function shiftDataset(dataset, hours) {
    let shifted = [];
    
    dataset.forEach(item => {
        // item.y[0] is start, item.y[1] is end
        let start = item.y[0] + hours;
        let end = item.y[1] + hours;

        // Helper to push a valid segment
        const pushSegment = (s, e) => {
            if (s < e) shifted.push({ x: item.x, y: [s, e] });
        };

        // If the record stays within the same 24-hour block
        if (start < 24 && end <= 24) {
            pushSegment(start, end);
        } 
        // If the record crosses into the NEXT day
        else if (start < 24 && end > 24) {
            pushSegment(start, 24);         // Part 1: Start to Midnight
            pushSegment(0, end % 24);       // Part 2: Midnight to End (on same X axis for simplicity)
        }
        // If the record started and ended in the NEXT day
        else if (start >= 24) {
            pushSegment(start % 24, end % 24);
        }
    });
    
    return shifted;
}

window.pivotChart = (tz) => {
    // 1. UI Update
    document.querySelectorAll('.tz-pill').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`btn-chart-${tz.toLowerCase()}`).classList.add('active');
    const dataElement = document.getElementById('chart-data');
    if (!dataElement) return;
    originalData = JSON.parse(dataElement.textContent);

    if (!originalData) return;

    // 2. Pivot Logic
    // We always calculate FROM the original PST data to avoid "drift" errors
    const shift = (tz === 'CST') ? 16 : 0;

    const shiftedData = {
        sleep: shiftDataset(originalData.sleep, shift),
        milk: shiftDataset(originalData.milk, shift),
        diaper: shiftDataset(originalData.diaper, shift)
    };

    renderChart(shiftedData);
};