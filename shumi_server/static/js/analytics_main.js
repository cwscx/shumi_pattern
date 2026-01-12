// static/js/analytics_main.js
import { getRollingMilkTotal } from './analytics_logic.js';

document.addEventListener('DOMContentLoaded', () => {
    const dataElement = document.getElementById('patterns-data');
    if (!dataElement) {
        console.error("Data element not found!");
        return;
    }

    try {
        const patterns = JSON.parse(dataElement.textContent);
        // We use 800 as default if milk_target from Django isn't easily accessible
        getRollingMilkTotal(patterns, 800); 
    } catch (e) {
        console.error("Failed to parse patterns data:", e);
    }
});