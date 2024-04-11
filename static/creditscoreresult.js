function drawPieChart() {
  let dataTable = [["Feature", "Percentage"]];
  shapValues.forEach((value) => {
      dataTable.push([value[0], value[1]]);
  });
  let data = google.visualization.arrayToDataTable(dataTable);

	let options = {
	  title: "Contributing Factors",
	  colors: ["#4A2626", "#FF6A69", "#223130", "#394645", "#223130"],
	  fontName: "Montserrat",
	  fontSize: 12,
	  titleTextStyle: {
		color: "black",
		fontName: "Montserrat",
		fontSize: 20,
	  },
	  legend: "none",
	  chartArea: { width: "100%", height: "100%" },
	};
	let chart = new google.visualization.PieChart(
	  document.getElementById("piechart")
	);
	chart.draw(data, options);
  }

  // For the Line Graph
 function drawLineChart() {
    let data = new google.visualization.DataTable();
    data.addColumn("number", "Year");
    data.addColumn("number", "Score");

    // Check if forecastScores is defined and use it; otherwise, fallback to default data
    if (typeof forecastScores !== 'undefined' && forecastScores.length > 0) {
        // Dynamically add rows from forecastScores
        forecastScores.forEach((score, index) => {
            data.addRow([index, score]);
        });
    } else {
        // Default data if forecastScores isn't available
        data.addRows([
            [0, 0],
            [1, 300],
            [2, 350],
            [3, 450],
            [4, 500],
            [5, 550],
        ]);
    }

    let options = {
        hAxis: {
            title: "Progression",
        },
        vAxis: {
            title: "Credit Score",
        },
        titleTextStyle: {
            color: "black",
            fontName: "Montserrat",
            fontSize: 20,
        },
        chartArea: { width: "80%", height: "80%" },
    };
    let chart = new google.visualization.LineChart(document.getElementById("line_chart_element"));
    chart.draw(data, options);
}


  // Load the charts
  const initCharts = () => {
	google.charts.load("current", { packages: ["corechart"] });
	google.charts.setOnLoadCallback(drawPieChart);
	google.charts.setOnLoadCallback(drawLineChart);
  };

  
  // Initialize the charts and other functions when the window loads
  window.onload = function () {
	initCharts();

	// Custom JS for heading css
	let elem = document.getElementById("scoreText");
	elem.classList.add("shown");

	// Custom JS for responsive - Redraw/Regenerate charts
	window.addEventListener("resize", function () {
	  drawPieChart();
	  drawLineChart();
	});
  };
