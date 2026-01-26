/**
 * France Property Prices - Mapbox GL JS Implementation
 * Interactive choropleth map with zoom-based aggregation levels (6 levels)
 * FastAPI Backend with Spatial Filtering
 */

// Validate Mapbox token
if (!MAPBOX_TOKEN || MAPBOX_TOKEN === 'your_mapbox_token_here') {
    alert('Please set your Mapbox access token in the .env file');
}

mapboxgl.accessToken = MAPBOX_TOKEN;

// Initialize map
const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/light-v11',
    center: [2.3, 46.6],
    zoom: 6,
    minZoom: 4,
    maxZoom: 16  // Increased for parcel level
});

// State
let currentLevel = 'region';
let currentData = null;
let isLoading = false;

/**
 * Determine aggregation level from zoom
 */
function getLevelFromZoom(zoom) {
    if (zoom < 6) return 'country';
    if (zoom < 7.5) return 'region';
    if (zoom < 9.5) return 'departement';
    if (zoom < 11.5) return 'commune';
    if (zoom < 13) return 'postcode';
    return 'parcel';  // Building plots at highest zoom
}

/**
 * Format number with thousand separators
 */
function formatNumber(num) {
    return new Intl.NumberFormat('fr-FR').format(Math.round(num));
}

/**
 * Format currency
 */
function formatCurrency(num) {
    return new Intl.NumberFormat('fr-FR', {
        style: 'currency',
        currency: 'EUR',
        maximumFractionDigits: 0
    }).format(num);
}

/**
 * Show loading indicator
 */
function showLoading() {
    isLoading = true;
    document.getElementById('loading').style.display = 'flex';
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    isLoading = false;
    document.getElementById('loading').style.display = 'none';
}

/**
 * Load price data from FastAPI with bounding box
 */
async function loadPriceData(level) {
    if (isLoading) {
        console.log('Already loading data, skipping...');
        return null;
    }
    
    showLoading();
    
    try {
        console.log(`Loading data for level: ${level}`);
        
        const bounds = map.getBounds();
        const bbox = `${bounds.getWest()},${bounds.getSouth()},${bounds.getEast()},${bounds.getNorth()}`;
        
        console.log(`Bounding box: ${bbox}`);
        
        const url = `/api/prices/${level}?bbox=${encodeURIComponent(bbox)}`;
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        currentData = data;
        
        console.log(`Loaded ${data.features.length} features for ${level} in current viewport`);
        
        document.getElementById('current-level').textContent = 
            level.charAt(0).toUpperCase() + level.slice(1);
        document.getElementById('region-count').textContent = 
            formatNumber(data.features.length);
        
        return data;
        
    } catch (error) {
        console.error('Error loading data:', error);
        alert(`Error loading ${level} data: ${error.message}`);
        return null;
    } finally {
        hideLoading();
    }
}

/**
 * Create strictly ascending color stops
 * Handles edge case where all prices are the same
 */
function createColorStops(prices, level) {
    const minPrice = prices[0];
    const maxPrice = prices[prices.length - 1];
    
    // Edge case: all prices are the same
    if (minPrice === maxPrice) {
        return [
            Math.max(0, Math.floor(minPrice * 0.8)),
            Math.floor(minPrice),
            Math.floor(minPrice * 1.2)
        ];
    }
    
    // Normal case: calculate percentiles
    const q25 = prices[Math.floor(prices.length * 0.25)];
    const median = prices[Math.floor(prices.length * 0.50)];
    const q75 = prices[Math.floor(prices.length * 0.75)];
    
    let rawStops;
    
    if (level === 'parcel') {
        // Parcels: use finer granularity
        const p10 = prices[Math.floor(prices.length * 0.10)];
        const p30 = prices[Math.floor(prices.length * 0.30)];
        const p50 = prices[Math.floor(prices.length * 0.50)];
        const p70 = prices[Math.floor(prices.length * 0.70)];
        const p90 = prices[Math.floor(prices.length * 0.90)];
        rawStops = [p10, p30, p50, p70, p90];
    } else if (level === 'postcode') {
        const p10 = prices[Math.floor(prices.length * 0.10)];
        const p30 = prices[Math.floor(prices.length * 0.30)];
        const p50 = prices[Math.floor(prices.length * 0.50)];
        const p70 = prices[Math.floor(prices.length * 0.70)];
        const p90 = prices[Math.floor(prices.length * 0.90)];
        rawStops = [p10, p30, p50, p70, p90];
    } else if (level === 'commune') {
        rawStops = [Math.max(500, minPrice), q25, median, q75, Math.min(15000, maxPrice)];
    } else {
        rawStops = [1000, 2000, 3000, 4500, 7000];
    }
    
    // Round and deduplicate
    let stops = [...new Set(rawStops.map(v => Math.floor(v)))].sort((a, b) => a - b);
    
    // Ensure strictly ascending by adding increments to duplicates
    const finalStops = [];
    let lastValue = -Infinity;
    
    for (let i = 0; i < stops.length; i++) {
        let value = stops[i];
        if (value <= lastValue) {
            value = lastValue + 10;
        }
        finalStops.push(value);
        lastValue = value;
    }
    
    // Ensure at least 2 different stops
    if (finalStops.length < 2) {
        finalStops.push(Math.floor(minPrice), Math.floor(maxPrice) + 1);
    }
    
    // Final dedup and sort
    const result = [...new Set(finalStops)].sort((a, b) => a - b);
    
    // If still only 1 unique value, force a range
    if (result.length === 1) {
        return [result[0], result[0] + 100];
    }
    
    return result;
}

/**
 * Update map layer with new data
 */
function updateMapLayer(data, level) {
    console.log(`Updating map layer for ${level} with ${data.features.length} features`);
    
    // Remove existing layers
    ['price-layer', 'price-outline', 'price-points'].forEach(layerId => {
        if (map.getLayer(layerId)) {
            map.removeLayer(layerId);
        }
    });
    
    if (map.getSource('prices')) {
        map.removeSource('prices');
    }
    
    // Add new source
    map.addSource('prices', {
        type: 'geojson',
        data: data
    });
    
    // Calculate price statistics
    const prices = data.features
        .map(f => f.properties.median_price)
        .filter(p => p > 0)
        .sort((a, b) => a - b);
    
    if (prices.length === 0) {
        console.error('No valid price data');
        return;
    }
    
    const minPrice = prices[0];
    const maxPrice = prices[prices.length - 1];
    const q25 = prices[Math.floor(prices.length * 0.25)];
    const median = prices[Math.floor(prices.length * 0.50)];
    const q75 = prices[Math.floor(prices.length * 0.75)];
    
    console.log(`Price range: €${minPrice.toFixed(0)} - €${maxPrice.toFixed(0)}`);
    console.log(`Quartiles: Q25=€${q25.toFixed(0)}, Median=€${median.toFixed(0)}, Q75=€${q75.toFixed(0)}`);
    
    // Create guaranteed ascending color stops
    const colorStops = createColorStops(prices, level);
    console.log('Color stops (guaranteed ascending):', colorStops);
    
    // Build color expression
    const colorExpression = ['interpolate', ['linear'], ['get', 'median_price']];
    
    // Color schemes by level
    const colors = level === 'parcel'
        ? ['#fff7bc', '#fee391', '#fec44f', '#fe9929', '#d95f0e']  // Yellow/orange for parcels
        : level === 'postcode' 
        ? ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']  // Blue for postcodes
        : ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'];  // Red for others
    
    // Add color stops (use up to 5 colors)
    const numStops = Math.min(colorStops.length, colors.length);
    for (let i = 0; i < numStops; i++) {
        colorExpression.push(colorStops[i], colors[i]);
    }
    
    // Add fill layer
    map.addLayer({
        id: 'price-layer',
        type: 'fill',
        source: 'prices',
        paint: {
            'fill-color': colorExpression,
            'fill-opacity': ['interpolate', ['linear'], ['zoom'], 6, 0.65, 9, 0.70, 11, 0.75, 13, 0.85, 15, 0.90]
        }
    });
    
    // Add outline layer with varying widths
    const outlineWidth = level === 'parcel'
        ? ['interpolate', ['linear'], ['zoom'], 13, 0.3, 14, 0.6, 16, 1.2]  // Thin for parcels
        : level === 'postcode' 
        ? ['interpolate', ['linear'], ['zoom'], 11.5, 0.5, 13, 1.0, 14, 1.5]
        : ['interpolate', ['linear'], ['zoom'], 4, 0.3, 8, 0.8, 10, 1.2, 14, 1.8];
    
    map.addLayer({
        id: 'price-outline',
        type: 'line',
        source: 'prices',
        paint: {
            'line-color': '#ffffff',
            'line-width': outlineWidth,
            'line-opacity': 0.8
        }
    });
    
    console.log('Map layer updated successfully');
    updateLegend(colorStops, level);
}

/**
 * Update legend
 */
function updateLegend(colorStops, level) {
    if (!colorStops || colorStops.length < 2) {
        console.warn('Invalid color stops for legend');
        return;
    }
    
    // Get 5 representative stops
    const step = Math.max(1, Math.floor((colorStops.length - 1) / 4));
    const legendStops = [
        colorStops[0],
        colorStops[Math.min(step, colorStops.length - 1)],
        colorStops[Math.min(step * 2, colorStops.length - 1)],
        colorStops[Math.min(step * 3, colorStops.length - 1)],
        colorStops[colorStops.length - 1]
    ];
    
    document.getElementById('range-1').textContent = `< €${formatNumber(legendStops[1])}`;
    document.getElementById('range-2').textContent = `€${formatNumber(legendStops[1])} - €${formatNumber(legendStops[2])}`;
    document.getElementById('range-3').textContent = `€${formatNumber(legendStops[2])} - €${formatNumber(legendStops[3])}`;
    document.getElementById('range-4').textContent = `€${formatNumber(legendStops[3])} - €${formatNumber(legendStops[4])}`;
    document.getElementById('range-5').textContent = `> €${formatNumber(legendStops[4])}`;
    
    const legendColors = document.querySelectorAll('.legend-color');
    
    if (level === 'parcel') {
        legendColors[0].style.background = '#fff7bc';
        legendColors[1].style.background = '#fee391';
        legendColors[2].style.background = '#fec44f';
        legendColors[3].style.background = '#fe9929';
        legendColors[4].style.background = '#d95f0e';
    } else if (level === 'postcode') {
        legendColors[0].style.background = '#ffffcc';
        legendColors[1].style.background = '#a1dab4';
        legendColors[2].style.background = '#41b6c4';
        legendColors[3].style.background = '#2c7fb8';
        legendColors[4].style.background = '#253494';
    } else {
        legendColors[0].style.background = '#fee5d9';
        legendColors[1].style.background = '#fcae91';
        legendColors[2].style.background = '#fb6a4a';
        legendColors[3].style.background = '#de2d26';
        legendColors[4].style.background = '#a50f15';
    }
    
    const noteElement = document.getElementById('legend-note');
    if (noteElement) {
        if (level === 'parcel') {
            noteElement.textContent = `Yellow polygons - Building plots (zoom ${map.getZoom().toFixed(1)})`;
        } else if (level === 'postcode') {
            noteElement.textContent = `Blue polygons - Postcode level (zoom ${map.getZoom().toFixed(1)})`;
        } else if (level === 'commune') {
            noteElement.textContent = `Red polygons - Commune level (zoom ${map.getZoom().toFixed(1)})`;
        } else {
            noteElement.textContent = 'National price scale';
        }
    }
}

/**
 * Create popup content with confidence scoring
 */
function createPopupContent(properties) {
    const name = properties.name || properties.code;
    const median = formatCurrency(properties.median_price);
    const weighted = formatCurrency(properties.weighted_price);
    const count = formatNumber(properties.transaction_count);
    const range = `${formatCurrency(properties.lower_bound)} - ${formatCurrency(properties.upper_bound)}`;
    
    // Confidence display
    const confidence = properties.confidence_score || 50;
    const quality = properties.estimate_quality || 'Medium';
    
    // Star rating based on confidence
    const starCount = Math.floor(confidence / 20);
    const stars = '⭐'.repeat(Math.max(1, starCount));
    
    // Color coding by confidence
    let confidenceColor, confidenceIcon, confidenceText;
    if (confidence >= 85) {
        confidenceColor = '#27ae60';  // Green
        confidenceIcon = '✓';
        confidenceText = 'Very Reliable';
    } else if (confidence >= 70) {
        confidenceColor = '#2ecc71';  // Light green
        confidenceIcon = '✓';
        confidenceText = 'Reliable';
    } else if (confidence >= 55) {
        confidenceColor = '#f39c12';  // Orange
        confidenceIcon = '!';
        confidenceText = 'Moderate';
    } else if (confidence >= 35) {
        confidenceColor = '#e67e22';  // Dark orange
        confidenceIcon = '⚠';
        confidenceText = 'Limited Data';
    } else {
        confidenceColor = '#e74c3c';  // Red
        confidenceIcon = '⚠';
        confidenceText = 'Insufficient Data';
    }
    
    // Volatility display
    const stdDev = properties.std_dev || 0;
    const volatilityPercent = median > 0 ? ((stdDev / properties.median_price) * 100).toFixed(1) : 0;
    
    // Days since last transaction
    let freshnessText = 'Unknown';
    if (properties.last_transaction_date) {
        const lastDate = new Date(properties.last_transaction_date);
        const daysSince = Math.floor((new Date() - lastDate) / (1000 * 60 * 60 * 24));
        
        if (daysSince <= 30) {
            freshnessText = `${daysSince} days ago`;
        } else if (daysSince <= 90) {
            freshnessText = `${Math.floor(daysSince / 30)} months ago`;
        } else {
            freshnessText = `${Math.floor(daysSince / 30)} months ago`;
        }
    }
    
    return `
        <div class="popup-content">
            <h3 style="margin-bottom: 12px; color: #2c3e50; font-size: 18px;">${name}</h3>
            
            <!-- Confidence Badge -->
            <div style="background: ${confidenceColor}; color: white; padding: 10px 12px; border-radius: 6px; margin-bottom: 15px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 4px;">
                    ${confidenceIcon} ${quality} Estimate ${stars}
                </div>
                <div style="font-size: 12px; opacity: 0.95;">
                    Confidence Score: ${confidence}/100 • ${confidenceText}
                </div>
            </div>
            
            <!-- Price Estimates -->
            <div style="background: #f8f9fa; padding: 12px; border-radius: 6px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                    <span style="font-size: 13px; color: #7f8c8d;">Best Estimate:</span>
                    <span style="font-size: 20px; font-weight: bold; color: #2c3e50;">${median}/m²</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                    <span style="font-size: 12px; color: #7f8c8d;">Confidence Interval:</span>
                    <span style="font-size: 14px; font-weight: 600; color: #34495e;">${range}/m²</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-size: 12px; color: #7f8c8d;">Time-Weighted:</span>
                    <span style="font-size: 14px; color: #3498db;">${weighted}/m²</span>
                </div>
            </div>
            
            <!-- Reliability Factors -->
            <div style="border-top: 1px solid #ecf0f1; padding-top: 10px; margin-top: 10px;">
                <div style="font-size: 11px; font-weight: 600; color: #7f8c8d; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Reliability Factors
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px;">
                    <div>
                        <span style="color: #95a5a6;">📊 Volume:</span><br>
                        <span style="font-weight: 600; color: #2c3e50;">${count} transactions</span>
                    </div>
                    <div>
                        <span style="color: #95a5a6;">📈 Volatility:</span><br>
                        <span style="font-weight: 600; color: #2c3e50;">±${volatilityPercent}%</span>
                    </div>
                    <div style="grid-column: 1 / -1;">
                        <span style="color: #95a5a6;">🕐 Last Transaction:</span><br>
                        <span style="font-weight: 600; color: #2c3e50;">${freshnessText}</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * Reset map view
 */
function resetView() {
    map.flyTo({
        center: [2.3, 46.6],
        zoom: 6,
        duration: 1500
    });
}

/**
 * Handle zoom change
 */
async function handleZoomChange() {
    const zoom = map.getZoom();
    const newLevel = getLevelFromZoom(zoom);
    
    document.getElementById('current-zoom').textContent = zoom.toFixed(1);
    console.log(`Zoom: ${zoom.toFixed(1)}, Current level: ${currentLevel}, New level: ${newLevel}`);
    
    if (newLevel !== currentLevel) {
        console.log(`Level changed from ${currentLevel} to ${newLevel}, reloading data...`);
        currentLevel = newLevel;
        
        const data = await loadPriceData(currentLevel);
        if (data && data.features.length > 0) {
            updateMapLayer(data, currentLevel);
        } else {
            console.warn(`No data returned for ${currentLevel}`);
        }
    }
}

/**
 * Add event listeners
 */
function addLayerInteractions() {
    map.on('mousemove', 'price-layer', (e) => {
        map.getCanvas().style.cursor = 'pointer';
        
        if (e.features.length > 0) {
            map.setPaintProperty('price-layer', 'fill-opacity', [
                'case',
                ['==', ['get', 'code'], e.features[0].properties.code],
                0.95,
                ['interpolate', ['linear'], ['zoom'], 6, 0.65, 9, 0.70, 11, 0.75, 13, 0.85, 15, 0.90]
            ]);
        }
    });
    
    map.on('mouseleave', 'price-layer', () => {
        map.getCanvas().style.cursor = '';
        if (map.getLayer('price-layer')) {
            map.setPaintProperty('price-layer', 'fill-opacity', 
                ['interpolate', ['linear'], ['zoom'], 6, 0.65, 9, 0.70, 11, 0.75, 13, 0.85, 15, 0.90]
            );
        }
    });
    
    map.on('click', 'price-layer', (e) => {
        const properties = e.features[0].properties;
        new mapboxgl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(createPopupContent(properties))
            .addTo(map);
    });
}

/**
 * Initialize map
 */
map.on('load', async function() {
    console.log('='.repeat(60));
    console.log('France Property Prices - Map Initialized');
    console.log('='.repeat(60));
    console.log('6 Aggregation Levels:');
    console.log('  Country (zoom < 6)');
    console.log('  Region (zoom 6-7.5)');
    console.log('  Département (zoom 7.5-9.5)');
    console.log('  Commune (zoom 9.5-11.5)');
    console.log('  Postcode (zoom 11.5-13)');
    console.log('  Building Plots (zoom 13+)');
    console.log('FastAPI Backend with Spatial Filtering');
    console.log('API Documentation: http://localhost:8080/docs');
    console.log('='.repeat(60));
    
    const data = await loadPriceData(currentLevel);
    if (data && data.features.length > 0) {
        updateMapLayer(data, currentLevel);
        addLayerInteractions();
    } else {
        console.error('Failed to load initial data');
        alert('Failed to load map data. Please check console and verify API is running.');
    }
    
    let zoomTimeout, lastZoom = map.getZoom();
    
    map.on('zoomend', () => {
        const newZoom = map.getZoom();
        console.log(`Zoom ended at: ${newZoom.toFixed(1)}`);
        
        if (Math.abs(newZoom - lastZoom) > 0.5) {
            clearTimeout(zoomTimeout);
            zoomTimeout = setTimeout(() => {
                handleZoomChange();
                lastZoom = newZoom;
            }, 300);
        }
    });
    
    let moveTimeout;
    map.on('moveend', () => {
        if (currentLevel === 'commune' || currentLevel === 'postcode' || currentLevel === 'parcel') {
            clearTimeout(moveTimeout);
            moveTimeout = setTimeout(async () => {
                console.log('Map moved, reloading data for current viewport...');
                const data = await loadPriceData(currentLevel);
                if (data && data.features.length > 0) {
                    updateMapLayer(data, currentLevel);
                }
            }, 500);
        }
    });
});

map.addControl(new mapboxgl.NavigationControl(), 'top-right');
map.addControl(new mapboxgl.FullscreenControl(), 'top-right');