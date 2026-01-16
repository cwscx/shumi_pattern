// static/js/analytics_main.js
import { getRollingMilkTotal, initFormulaTrend, getRollingSleepTotal, initBioClockChart } from './analytics.js';

document.addEventListener('DOMContentLoaded', () => {
    const dataElement = document.getElementById('patterns-data');
    if (!dataElement) {
        console.error("Data element not found!");
        return;
    }

    try {
        console.log("analytics_main triggers analytics logic");
        const patterns = JSON.parse(dataElement.textContent);
        // We use 800 as default if milk_target from Django isn't easily accessible
        getRollingMilkTotal(patterns, 800); 
        initFormulaTrend(patterns);
        getRollingSleepTotal(patterns);
        initBioClockChart(patterns);
    } catch (e) {
        console.error("Failed to parse patterns data:", e);
    }
});