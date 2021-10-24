const canvas = d3.select(".canva")
const h = 800
const w = 800
const margin = {top: 10, right: 30, bottom: 30, left: 60},
    width = w - margin.left - margin.right,
    height = h - margin.top - margin.bottom;

// Create the svg to be displayed in div class canva
const svg = canvas.append("svg")
    .attr("width", w + margin.left + margin.right)
    .attr("height", h + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
          `translate(${margin.left}, ${margin.top})`);

// Make several layers
const layers1 = [
  new deck.ScatterplotLayer({
    // data
    getFillColor: d => d.color,
    getRadius: d => d.radius
  }),

]

// https://gisgeography.com/wgs84-world-geodetic-system/
// https://gisgeography.com/latitude-longitude-coordinates/
    new deck.DeckGL({
      mapStyle: 'https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json',
      initialViewState: {
        // Center Moscow
        //longitude: 37.6155,
        //latitude: 55.75222,
        longitude: 37.741109,
        latitude: 55.59516,
        zoom: 15
      },
      controller: true,
      layers: [
        new deck.ScatterplotLayer({
          data: [
            {position: [37.741109, 55.59516], color: [255, 0, 0], radius: 100}
          ],
          getFillColor: d => d.color,
          getRadius: d => d.radius
        })
      ]
    });
