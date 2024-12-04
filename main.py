from flask import Flask, request
from flask_cors import CORS
import json

from config import NRTLParams, R
from solver import NRTL

from encoder import NumpyArrayEncoder

app=Flask(__name__)
CORS(app)

@app.route("/api/sle", methods=["POST"])
def solve_sle():
    try:
        data = request.get_json()
        inputs = NRTLParams(
            x=[data.get('x1'), data.get('x2')],
            T=data.get('T'),
            P=data.get('P'),
            delta_g12=data.get('delta_g12'),
            delta_g21=data.get('delta_g21'),
            alpha=data.get('alpha'),
            Tm=[data.get('Tm1'), data.get('Tm2')],
            delta_H_fus=[data.get('delta_H_fus1'), data.get('delta_H_fus2')]
        )
        response = NRTL(inputs)
        result = NRTL(inputs)
        response = {
            'code': 0,
            'message': 'OK',
            'solution': {
                'phase': result.phase.value,
                'boundary': result.boundary,
                'xe': result.xe,
                'Te': result.Te,
                'xS': result.xS,
                'xL': result.xL
            }
        }
        return json.dumps(response, cls=NumpyArrayEncoder), 200
    except Exception as e:
        return json.dumps({
                'code': 1,
                'message': str(e),
                'solution': {}
            }), 200

if __name__ == '__main__':
    #p-toluenesulfonamide (1) + benzamide (2)
    # inputs = NRTLParams(
    #     x=[0.2,0.8],
    #     T=500,
    #     P=1,
    #     delta_g12=253.55 * R,
    #     delta_g21=-492.79 * R,
    #     alpha=0.3,
    #     Tm=[410.2, 400],
    #     delta_H_fus=[22.47*(10**3), 21.3*(10**3)],
    # )
    app.run(host='0.0.0.0', port=8000)

