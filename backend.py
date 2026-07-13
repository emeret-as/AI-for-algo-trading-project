import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.orchestration.jobs import run_full_pipeline
from src.orchestration.replay import replay_date_range

app = FastAPI(title='Volatility Pipeline API')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class RunPipelineRequest(BaseModel):
    date: str
    underlying: str = 'SPY'
    spot: float = 450.0
    regime: str = 'normal'
    seed: int = 42
    num_strikes: int = 21
    num_monthly: int = 4
    include_weeklies: bool = True
    dry_run: bool = True


class ReplayRequest(BaseModel):
    start_date: str
    end_date: str
    underlying: str = 'SPY'
    spot: float = 450.0
    regime: str = 'normal'
    seed: int = 42
    num_strikes: int = 21
    num_monthly: int = 4
    include_weeklies: bool = True
    dry_run: bool = True


@app.post('/api/run-pipeline')
def run_pipeline(req: RunPipelineRequest):
    try:
        result = run_full_pipeline(
            trade_date_str=req.date,
            underlying_symbol=req.underlying,
            spot=req.spot,
            data_dir='data',
            config={
                'regime': req.regime,
                'seed': req.seed,
                'num_strikes': req.num_strikes,
                'num_monthly': req.num_monthly,
                'include_weeklies': req.include_weeklies,
            },
            dry_run=req.dry_run,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post('/api/replay')
def run_replay(req: ReplayRequest):
    try:
        results = replay_date_range(
            req.start_date,
            req.end_date,
            req.underlying,
            data_dir='data',
            config={
                'regime': req.regime,
                'seed': req.seed,
                'num_strikes': req.num_strikes,
                'num_monthly': req.num_monthly,
                'include_weeklies': req.include_weeklies,
            },
        )
        return {'results': results, 'count': len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get('/api/config')
def get_config():
    config_path = Path(ROOT) / 'configs' / 'default.yaml'
    return {'config': config_path.read_text(encoding='utf-8')}
