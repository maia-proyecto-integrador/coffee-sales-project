import { Component, signal, ViewChild, ElementRef, AfterViewInit } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Chart, ChartConfiguration, registerables } from 'chart.js';

Chart.register(...registerables);

interface ForecastData {
  type: string;
  sales: number;
  forecast: number;
}

@Component({
  selector: 'app-root',
  imports: [FormsModule, CommonModule],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App implements AfterViewInit {
  @ViewChild('chartCanvas', { static: false }) chartCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('forecastChartCanvas', { static: false }) forecastChartCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('validationChartCanvas', { static: false }) validationChartCanvas!: ElementRef<HTMLCanvasElement>;
  
  protected readonly title = signal('coffee-sales');
  
  selectedCoffeeType = 'espresso';
  selectedDateRange = 'apr1-apr21';
  
  selectedCoffeeTypeForecast = 'default';
  selectedDateRangeForecast = 'apr1-apr21';
  
  totalSales = 9450;
  nextPeriod = 3840;
  percentChange = 9.8;
  
  forecastTotalSales = 12340;
  forecastPercentChange = 15.2;
  
  // Restock data
  nextRestockEspresso = 'En 2 días';
  restockQuantityEspresso = 25;
  currentStockEspresso = 3.5;
  
  nextRestockMilk = 'En 5 días';
  restockQuantityMilk = 50;
  currentStockMilk = 12;
  
  nextRestockCocoa = 'En 12 días';
  restockQuantityCocoa = 8;
  currentStockCocoa = 4.2;
  
  nextRestockSugar = 'En 15 días';
  restockQuantitySugar = 15;
  currentStockSugar = 8.5;
  
  // Validation metrics
  forecastAccuracy = 87.3;
  averageError = 12.7;
  reliabilityScore = 'Alta';
  
  forecastData: ForecastData[] = [
    { type: 'Espresso', sales: 4500, forecast: 1800 },
    { type: 'Americano', sales: 3100, forecast: 1200 },
    { type: 'Cappuccino', sales: 1850, forecast: 840 }
  ];

  private chart: Chart | null = null;
  private forecastChart: Chart | null = null;
  private validationChart: Chart | null = null;

  ngAfterViewInit() {
    this.createChart();
    this.createForecastChart();
    this.createValidationChart();
  }

  onCoffeeTypeChange() {
    this.updateData();
  }

  onDateRangeChange() {
    this.updateData();
  }

  onCoffeeTypeForecastChange() {
    this.updateForecastData();
  }

  onDateRangeForecastChange() {
    this.updateForecastData();
  }

  private updateData() {
    // Simulate data changes based on selections
    if (this.selectedCoffeeType === 'espresso') {
      this.totalSales = 9450;
      this.nextPeriod = 3840;
      this.percentChange = 9.8;
    } else if (this.selectedCoffeeType === 'americano') {
      this.totalSales = 7200;
      this.nextPeriod = 2900;
      this.percentChange = 7.2;
    } else {
      this.totalSales = 5100;
      this.nextPeriod = 2100;
      this.percentChange = 12.5;
    }
    
    this.updateChart();
  }

  private createChart() {
    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) return;

    const config: ChartConfiguration = {
      type: 'line',
      data: {
        labels: ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
        datasets: [{
          label: 'Ventas Históricas',
          data: [245, 189, 267, 312, 398, 445, 321],
          borderColor: '#8B4513',
          backgroundColor: 'rgba(139, 69, 19, 0.1)',
          borderWidth: 3,
          pointRadius: 6,
          pointBackgroundColor: '#8B4513',
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          tension: 0.4,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          }
        },
        scales: {
          x: {
            grid: {
              display: true,
              color: '#f0f0f0'
            },
            border: {
              display: false
            },
            ticks: {
              color: '#666',
              font: {
                size: 12,
                weight: 500
              }
            }
          },
          y: {
            beginAtZero: true,
            max: 500,
            grid: {
              display: true,
              color: '#f0f0f0'
            },
            border: {
              display: false
            },
            ticks: {
              color: '#666',
              font: {
                size: 12
              }
            },
            title: {
              display: true,
              text: 'Cantidad',
              color: '#666',
              font: {
                size: 14,
                weight: 600
              }
            }
          }
        },
        interaction: {
          intersect: false,
          mode: 'index'
        }
      }
    };

    this.chart = new Chart(ctx, config);
  }

  private updateChart() {
    if (!this.chart) return;
    
    // Update chart data based on selected coffee type
    let weeklyData: number[];
    
    if (this.selectedCoffeeType === 'espresso') {
      weeklyData = [180, 145, 198, 225, 289, 312, 234];
    } else if (this.selectedCoffeeType === 'americano') {
      weeklyData = [210, 167, 223, 278, 334, 378, 289];
    } else if (this.selectedCoffeeType === 'cappuccino') {
      weeklyData = [156, 134, 178, 201, 245, 267, 198];
    } else if (this.selectedCoffeeType === 'latte') {
      weeklyData = [189, 156, 201, 234, 278, 301, 245];
    } else if (this.selectedCoffeeType === 'cortado') {
      weeklyData = [123, 98, 134, 156, 189, 201, 167];
    } else if (this.selectedCoffeeType === 'hot-chocolate') {
      weeklyData = [98, 89, 112, 134, 167, 189, 145];
    } else if (this.selectedCoffeeType === 'cocoa') {
      weeklyData = [67, 56, 78, 89, 112, 134, 98];
    } else {
      // Default - All beverages combined
      weeklyData = [245, 189, 267, 312, 398, 445, 321];
    }
    
    this.chart.data.datasets[0].data = weeklyData;
    this.chart.update();
  }

  private updateForecastData() {
    // Simulate forecast data changes based on selections
    if (this.selectedCoffeeTypeForecast === 'espresso') {
      this.forecastTotalSales = 11200;
      this.forecastPercentChange = 18.5;
    } else if (this.selectedCoffeeTypeForecast === 'americano') {
      this.forecastTotalSales = 13400;
      this.forecastPercentChange = 22.1;
    } else if (this.selectedCoffeeTypeForecast === 'cappuccino') {
      this.forecastTotalSales = 8900;
      this.forecastPercentChange = 14.3;
    } else {
      this.forecastTotalSales = 12340;
      this.forecastPercentChange = 15.2;
    }
    
    this.updateForecastChart();
  }

  private createForecastChart() {
    const ctx = this.forecastChartCanvas.nativeElement.getContext('2d');
    if (!ctx) return;

    const config: ChartConfiguration = {
      type: 'line',
      data: {
        labels: ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
        datasets: [{
          label: 'Pronóstico de Ventas',
          data: [285, 220, 310, 365, 445, 510, 380],
          borderColor: '#90EE90',
          backgroundColor: 'rgba(144, 238, 144, 0.1)',
          borderWidth: 3,
          borderDash: [8, 4],
          pointRadius: 6,
          pointBackgroundColor: '#90EE90',
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          tension: 0.4,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          }
        },
        scales: {
          x: {
            grid: {
              display: true,
              color: '#f0f0f0'
            },
            border: {
              display: false
            },
            ticks: {
              color: '#666',
              font: {
                size: 12,
                weight: 500
              }
            }
          },
          y: {
            beginAtZero: true,
            max: 600,
            grid: {
              display: true,
              color: '#f0f0f0'
            },
            border: {
              display: false
            },
            ticks: {
              color: '#666',
              font: {
                size: 12
              }
            },
            title: {
              display: true,
              text: 'Cantidad',
              color: '#666',
              font: {
                size: 14,
                weight: 600
              }
            }
          }
        },
        interaction: {
          intersect: false,
          mode: 'index'
        }
      }
    };

    this.forecastChart = new Chart(ctx, config);
  }

  private updateForecastChart() {
    if (!this.forecastChart) return;
    
    // Update forecast chart data based on selected coffee type
    let forecastWeeklyData: number[];
    
    if (this.selectedCoffeeTypeForecast === 'espresso') {
      forecastWeeklyData = [210, 168, 230, 260, 335, 365, 275];
    } else if (this.selectedCoffeeTypeForecast === 'americano') {
      forecastWeeklyData = [245, 195, 260, 320, 385, 435, 335];
    } else if (this.selectedCoffeeTypeForecast === 'cappuccino') {
      forecastWeeklyData = [180, 155, 205, 235, 285, 310, 230];
    } else if (this.selectedCoffeeTypeForecast === 'latte') {
      forecastWeeklyData = [220, 180, 235, 270, 320, 350, 285];
    } else if (this.selectedCoffeeTypeForecast === 'cortado') {
      forecastWeeklyData = [145, 115, 155, 180, 220, 235, 195];
    } else if (this.selectedCoffeeTypeForecast === 'hot-chocolate') {
      forecastWeeklyData = [115, 105, 130, 155, 195, 220, 170];
    } else if (this.selectedCoffeeTypeForecast === 'cocoa') {
      forecastWeeklyData = [80, 65, 90, 105, 130, 155, 115];
    } else {
      // Default - All beverages combined forecast
      forecastWeeklyData = [285, 220, 310, 365, 445, 510, 380];
    }
    
    this.forecastChart.data.datasets[0].data = forecastWeeklyData;
    this.forecastChart.update();
  }

  private createValidationChart() {
    const ctx = this.validationChartCanvas.nativeElement.getContext('2d');
    if (!ctx) return;

    const config: ChartConfiguration = {
      type: 'line',
      data: {
        labels: ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
        datasets: [{
          label: 'Ventas Reales',
          data: [245, 189, 267, 312, 398, 445, 321],
          borderColor: '#8B4513',
          backgroundColor: 'rgba(139, 69, 19, 0.1)',
          borderWidth: 3,
          pointRadius: 6,
          pointBackgroundColor: '#8B4513',
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          tension: 0.4,
          fill: true
        }, {
          label: 'Pronóstico Anterior',
          data: [285, 220, 310, 365, 445, 510, 380],
          borderColor: '#90EE90',
          backgroundColor: 'rgba(144, 238, 144, 0.1)',
          borderWidth: 3,
          borderDash: [8, 4],
          pointRadius: 6,
          pointBackgroundColor: '#90EE90',
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          tension: 0.4,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          }
        },
        scales: {
          x: {
            grid: {
              display: true,
              color: '#f0f0f0'
            },
            border: {
              display: false
            },
            ticks: {
              color: '#666',
              font: {
                size: 12,
                weight: 500
              }
            }
          },
          y: {
            beginAtZero: true,
            max: 600,
            grid: {
              display: true,
              color: '#f0f0f0'
            },
            border: {
              display: false
            },
            ticks: {
              color: '#666',
              font: {
                size: 12
              }
            },
            title: {
              display: true,
              text: 'Cantidad',
              color: '#666',
              font: {
                size: 14,
                weight: 600
              }
            }
          }
        },
        interaction: {
          intersect: false,
          mode: 'index'
        }
      }
    };

    this.validationChart = new Chart(ctx, config);
  }
}