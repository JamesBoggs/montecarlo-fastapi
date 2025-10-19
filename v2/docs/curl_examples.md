# Copy-paste cURL examples

## rl-pricing
```bash
curl -sS -X POST $RL_URL/predict -H 'Content-Type: application/json' -d '{
  "params": {"explore": 0.1},
  "data": {"features": [[1.2,0.3,0.1]], "action_space": [0,1,2]}
}'
```

## garch
```bash
curl -sS -X POST $GARCH_URL/predict -H 'Content-Type: application/json' -d '{
  "data": {"returns": [-0.01,0.004,0.002,-0.006], "alpha": 0.95}
}'
```

## montecarlo
```bash
curl -sS -X POST $MC_URL/predict -H 'Content-Type: application/json' -d '{
  "data": {"mu": 0.08, "sigma": 0.2, "horizon": 10, "paths": 5000, "seed": 42}
}'
```

## forecast
```bash
curl -sS -X POST $FC_URL/predict -H 'Content-Type: application/json' -d '{
  "params": {"seasonal": "auto"},
  "data": {"series": [["2025-09-01",120.0],["2025-09-02",121.4]], "horizon": 14}
}'
```

## sentiment
```bash
curl -sS -X POST $SENT_URL/predict -H 'Content-Type: application/json' -d '{
  "data": {"texts": ["Fed hints at pause", "Earnings miss worries investors"]}
}'
```

## elasticity
```bash
curl -sS -X POST $ELAS_URL/predict -H 'Content-Type: application/json' -d '{
  "data": {"prices": [10,11,12], "quantities": [100,92,85], "price": 12.5}
}'
```

## price-engine
```bash
curl -sS -X POST $PE_URL/predict -H 'Content-Type: application/json' -d '{
  "data": {"cost": 8.50, "base_price": 12.0, "features": [0.2, 1.0, 3.4]}
}'
```
