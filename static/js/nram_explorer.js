let isSimulationPaused = false;
const ws = new WebSocket(`ws://${window.location.host}/v1/ws/nram-explorer`);
const tooltip = d3.select('body').append('div').attr('class', 'tooltip').style('opacity', 0);

function toggleSection(sectionId) {
    const content = document.getElementById(sectionId);
    content.classList.toggle('active');
    const header = content.previousElementSibling;
    const toggle = header.querySelector('.toggle');
    toggle.textContent = content.classList.contains('active') ? '▼' : '▶';
}

function updateMemoryGrid(memoryState) {
    const grid = d3.select('#memory-grid');
    const cells = grid.selectAll('.memory-cell').data(memoryState);

    cells.enter()
        .append('div')
        .attr('class', 'memory-cell')
        .merge(cells)
        .style('background-color', d => d3.interpolateViridis(Math.abs(d)))
        .on('mouseover', function(event, d) {
            tooltip.transition().duration(200).style('opacity', .9);
            tooltip.html(`Value: ${d.toFixed(4)}`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
        })
        .on('mouseout', function() {
            tooltip.transition().duration(500).style('opacity', 0);
        });

    cells.exit().remove();
}

function updatePatternMatrix(patterns) {
    const svg = d3.select('#pattern-matrix');
    const width = 200;
    const height = 200;
    const cellSize = Math.min(width / patterns[0].length, height / patterns.length);

    svg.selectAll('*').remove();

    const matrix = svg.append('g')
        .attr('transform', 'translate(10,10)');

    patterns.forEach((row, i) => {
        row.forEach((value, j) => {
            matrix.append('rect')
                .attr('x', j * cellSize)
                .attr('y', i * cellSize)
                .attr('width', cellSize - 1)
                .attr('height', cellSize - 1)
                .attr('fill', d3.interpolateViridis(value))
                .on('mouseover', function(event) {
                    tooltip.transition().duration(200).style('opacity', .9);
                    tooltip.html(`Pattern Strength: ${value.toFixed(4)}`)
                        .style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 10) + 'px');
                })
                .on('mouseout', function() {
                    tooltip.transition().duration(500).style('opacity', 0);
                });
        });
    });
}

function updateStats(stats) {
    document.getElementById('consciousness-level').textContent = stats.consciousness_level;
    document.getElementById('pattern-intensity').textContent = 
        typeof stats.pattern_intensity === 'number' 
            ? stats.pattern_intensity.toFixed(4) 
            : stats.pattern_intensity;
}

let simulation;
function initializeNetworkVisualization() {
    const container = document.getElementById('network-visualization');
    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3.select('#network-visualization')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    simulation = d3.forceSimulation()
        .force('link', d3.forceLink().id(d => d.id).distance(50))
        .force('charge', d3.forceManyBody().strength(-100))
        .force('center', d3.forceCenter(width / 2, height / 2));

    return svg;
}

function updateNetworkVisualization(data, svg) {
    const nodes = data.nodes;
    const links = data.links;

    const link = svg.selectAll('.link')
        .data(links)
        .join('line')
        .attr('class', 'link')
        .style('stroke', '#666')
        .style('stroke-opacity', 0.6);

    const node = svg.selectAll('.node')
        .data(nodes)
        .join('circle')
        .attr('class', 'node')
        .attr('r', 5)
        .style('fill', '#4CAF50')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));

    simulation
        .nodes(nodes)
        .on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
        });

    simulation.force('link').links(links);
    simulation.alpha(1).restart();
}

function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
}

function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
}

function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
}

function pauseSimulation() {
    isSimulationPaused = !isSimulationPaused;
    const button = document.querySelector('button');
    button.textContent = isSimulationPaused ? 'Resume' : 'Pause';
}

function resetSimulation() {
    ws.send(JSON.stringify({action: 'reset'}));
}

const networkSvg = initializeNetworkVisualization();

async function updateConfigurationSuggestions() {
    try {
        const response = await fetch('/v1/nram/config/analyze');
        const data = await response.json();

        const suggestionsHtml = `
            <div class="suggestion-group">
                <h4>Performance Metrics</h4>
                <p>Pattern Intensity: ${data.performance_metrics.avg_pattern_intensity.toFixed(3)}</p>
                <p>State Stability: ${data.performance_metrics.state_stability.toFixed(3)}</p>
            </div>
            <div class="suggestion-group">
                <h4>Suggested Changes</h4>
                <pre>${JSON.stringify(data.suggestions, null, 2)}</pre>
            </div>
        `;

        document.getElementById('suggestions-content').innerHTML = suggestionsHtml;
    } catch (error) {
        console.error('Error fetching suggestions:', error);
    }
}

async function applyConfiguration() {
    try {
        const response = await fetch('/v1/nram/config/analyze');
        const data = await response.json();

        await fetch('/v1/nram/config/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data.suggestions.suggestions)
        });

        updateConfigurationSuggestions();
    } catch (error) {
        console.error('Error applying configuration:', error);
    }
}

async function resetConfiguration() {
    try {
        await fetch('/v1/nram/config/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                memory_size: 1024,
                entropy_factor: 0.3,
                consciousness_levels: {
                    baseline: 0.3,
                    aware: 0.5,
                    enlightened: 0.7,
                    transcendent: 0.9
                }
            })
        });

        updateConfigurationSuggestions();
    } catch (error) {
        console.error('Error resetting configuration:', error);
    }
}

ws.onmessage = function(event) {
    if (isSimulationPaused) return;
    const data = JSON.parse(event.data);
    updateMemoryGrid(data.memory_state);
    updatePatternMatrix(data.pattern_memory);
    updateStats({
        consciousness_level: data.consciousness_level,
        pattern_intensity: data.pattern_intensity
    });
    updateNetworkVisualization(data.network, networkSvg);
    updateConfigurationSuggestions();
};

toggleSection('memory');
toggleSection('stats');
