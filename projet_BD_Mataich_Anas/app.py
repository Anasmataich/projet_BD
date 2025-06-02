from flask import Flask, render_template, request
from recommender import advanced_recommend

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        country = request.form.get('country')
        ingredients = request.form.get('ingredients')
        nutriscore_min = request.form.get('nutriscore_min')
        nutriscore_max = request.form.get('nutriscore_max')  # pas utilisé ici mais possible à gérer
        packaging = request.form.get('packaging')  # non géré pour l’instant
        ecoscore = request.form.get('ecoscore')    # non géré pour l’instant
        allergens = request.form.getlist('allergens')

        # Simple gestion allergènes : si liste non vide, on active allergen_free
        allergen_free = len(allergens) > 0

        raw_results = advanced_recommend(
    country=country,
    ingredients=ingredients,
    nutriscore_min=nutriscore_min,  # Utilise nutriscore_min au lieu de nutriscore
    nutriscore_max=nutriscore_max,
    allergen_free=bool(allergens),
    top_n=10
)


        for product in raw_results:
            recommendations.append({
                'product_name': product.get('product_name', 'Inconnu'),
                'brands': product.get('brands', 'N/A'),
                'nutrition_grade_fr': product.get('nutrition_grade_fr', 'N/A'),
                'ingredients_text': product.get('ingredients_text', 'N/A'),
                'image_url': product.get('image_url'),
                'url': product.get('url', 'N/A')
            })

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
