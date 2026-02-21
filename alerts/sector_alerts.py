import json
import pandas as pd
from datetime import datetime

class SectorAlertSystem:
    def __init__(self, report_data):
        self.data = report_data
        self.alerts = []

    def analyze_sector_performance(self):
        """Identify sectors with significant moves (e.g. >3% weekly)."""
        sectors = self.data.get('sector_overview', [])
        if not sectors:
            return

        # Sort by 1W performance
        sorted_1w = sorted(sectors, key=lambda x: x['performance'].get('1W', 0), reverse=True)
        
        # Best Performer
        best = sorted_1w[0]
        perf_1w = best['performance'].get('1W', 0)
        
        if perf_1w > 2.0:
            self.alerts.append({
                "type": "SECTOR_GROWTH",
                "severity": "high" if perf_1w > 4.0 else "medium",
                "title": f"🚀 {best['sector']} Leading the Market",
                "message": f"{best['sector']} ({best['etf']}) is up {perf_1w:.2f}% this week.",
                "sector": best['sector']
            })
            
        # Worst Performer (if significant drop)
        worst = sorted_1w[-1]
        loss_1w = worst['performance'].get('1W', 0)
        
        if loss_1w < -2.0:
            self.alerts.append({
                "type": "SECTOR_DROP",
                "severity": "warning",
                "title": f"⚠️ {worst['sector']} Under Pressure",
                "message": f"{worst['sector']} ({worst['etf']}) is down {loss_1w:.2f}% this week.",
                "sector": worst['sector']
            })

    def analyze_volume_surges(self):
        """Identify sectors with unusual volume."""
        sectors = self.data.get('sector_overview', [])
        for s in sectors:
            vol_spike = s.get('volume_spike', 1.0)
            if vol_spike > 1.5:
                self.alerts.append({
                    "type": "VOLUME_SPIKE",
                    "severity": "high",
                    "title": f"📢 High Volume in {s['sector']}",
                    "message": f"Trading volume is {vol_spike:.1f}x the average. Watch for big moves.",
                    "sector": s['sector']
                })

    def find_top_sector_stocks(self):
        """Find standout stocks in the flagged sectors."""
        # Only look at sectors we already alerted on
        alerted_sectors = {a['sector'] for a in self.alerts if 'sector' in a}
        
        constituents = self.data.get('constituents', {})
        
        for sector in alerted_sectors:
            stocks = constituents.get(sector, [])
            if not stocks:
                continue
                
            # Filter for strong stocks in this sector
            # Criteria: Positive Trend + Positive Momentum Score
            valid_stocks = [
                s for s in stocks 
                if s['total_score'] > 60 and s['performance'].get('1W', 0) > 0
            ]
            
            # Sort by Total Score
            top_stocks = sorted(valid_stocks, key=lambda x: x['total_score'], reverse=True)[:3]
            
            if top_stocks:
                stock_list = ", ".join([f"{s['ticker']} ({s['market_cap_category']})" for s in top_stocks])
                self.alerts.append({
                    "type": "STOCK_PICKS",
                    "severity": "info",
                    "title": f"🔥 Top Picks in {sector}",
                    "message": f"Watch these leaders: {stock_list}",
                    "sector": sector
                })

    def generate_alerts(self):
        self.analyze_sector_performance()
        self.analyze_volume_surges()
        self.find_top_sector_stocks()
        
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "alerts": self.alerts
        }

if __name__ == "__main__":
    # Test
    with open("../detailed_sector_report.json", "r") as f:
        data = json.load(f)
    
    system = SectorAlertSystem(data)
    alerts = system.generate_alerts()
    print(json.dumps(alerts, indent=2))
