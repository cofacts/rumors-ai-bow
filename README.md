# Rumors-AI-template

## Use docker

```bash
docker run --env CFA_ACTION=register darkbtf/rumors-ai-bow
docker run --env CFA_ACTION=start --env CFA_API_KEY=mL3k7tY-7VblR2sjifKMZEri2pD0nX_slj3BwpDDrKg --env CFA_ID=600430f4a431ea627d06c00e darkbtf/rumors-ai-bow
```

## Use npm

```bash
CFA_ACTION=register python main.py
CFA_ACTION=start CFA_API_KEY=mL3k7tY-7VblR2sjifKMZEri2pD0nX_slj3BwpDDrKg CFA_ID=600430f4a431ea627d06c00e python main.py
```

## Train model

python train.py
