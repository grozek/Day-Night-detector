import React, { useState } from 'react';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState("");

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    setImage(URL.createObjectURL(file));
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResult(data.result);
  };

  return (
    <div
      key={result}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        minHeight: "100vh",
        fontFamily: "Arial, sans-serif",
        padding: "2rem",
        transition: "background 0.5s ease",
        background: result === "day"
          ? "linear-gradient(to bottom, #62c6eb, #ffffff)"
          : result === "night"
            ? "linear-gradient(to bottom, #000000, #222222)"
            : "linear-gradient(to bottom, #f9a825, #ffffff)",
        transition: "background 0.6s ease-in-out",
      }}>
      <h1 style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>
        Day or Night?
      </h1>

      {image && (
        <img
          src={image}
          alt="Preview"
          style={{
            maxWidth: "90%",
            maxHeight: "300px",
            margin: "1rem 0",
            borderRadius: "12px",
            boxShadow: "0 4px 10px rgba(0,0,0,0.2)",
            border: "10px solid",
            borderColor: result === "day" ? "#f9a825" : result === "night" ? "#0d47a1" : "#ccc"
          }}
        />
      )}

      <input
        type="file"
        accept="image/*"
        onChange={handleUpload}
        style={{
          marginBottom: "1rem",
          padding: "0.5rem",
          fontSize: "1rem",
          cursor: "pointer",
          border: "10px solid #f9a825",
          borderRadius: "6px",
          background: "white",
          width: "300px",
          height: "25px",
        }}
      />

      {result && (
        <h2 style={{
          fontSize: 80,
          fontWeight: 1000,
          textTransform: 'uppercase',
          fontFamily: 'Russo One',
          color: result === "day" ? "#f9a825" : "#0d47a1"
        }}>
          {result}
        </h2>
      )}
    </div>
  );


}

export default App;



// {result && (
//   <h2 style={{
//     color: result === "day" ? "#f9a825" : "#0d47a1",
//     fontSize: "1.5rem",
//     marginTop: "1rem",
//     marginBottom: "1rem",
//     padding: "0.5rem",
//     fontSize: "1rem",
//     cursor: "pointer",
//     border: "10px solid #f9a825",
//     borderRadius: "6px",
//     background: "white",
//     width: "300px",
//     height: "25px",
//   }}>
//     Result: {result}
//   </h2>
// )}