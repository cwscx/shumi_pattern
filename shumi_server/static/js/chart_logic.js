// static/js/chart_logic.js

export function initChart() {
    const dataElement = document.getElementById('chart-data');
    if (!dataElement) return;
    const rawData = JSON.parse(dataElement.textContent);

    const ctx = document.getElementById('routineChart').getContext('2d');
    
    new Chart(ctx, {
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